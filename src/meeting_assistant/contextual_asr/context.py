"""Bounded deterministic extraction. Documents are data, never model instructions."""

import csv
import hashlib
import io
import json
import re
import stat
from dataclasses import replace
from pathlib import Path

from meeting_assistant.grounding.glossary import load_glossary
from meeting_assistant.grounding.models import GlossaryEntry
from meeting_assistant.grounding.normalization import normalize

from .exceptions import InvalidContext
from .models import ContextSource, ContextTerm, MeetingContextPack, text


def sha256(data):
    return hashlib.sha256(data.encode("utf-8") if isinstance(data, str) else data).hexdigest()


def extract_terms(content, known_entries):
    """Precision-first known names, acronyms, CamelCase and model identifiers."""
    found = {}
    for entry in known_entries:
        if re.search(r"(?<!\w)" + re.escape(entry.canonical) + r"(?!\w)", content, re.I):
            found.setdefault(normalize(entry.canonical), entry.canonical)
    for match in re.finditer(
        r"\b(?:[A-Z]{2,10}|[A-Z][a-z]+(?:[A-Z][A-Za-z]+)+|[A-Za-z]+(?:-[A-Za-z0-9]+)+)\b",
        content,
    ):
        value = match[0]
        # Full sentences/commands and ordinary title case are deliberately not extracted.
        found.setdefault(normalize(value), value)
    return tuple(found.values())[:256]


def build_meeting_context(
    *,
    title=None,
    agenda=(),
    description=None,
    terms=(),
    participants=(),
    documents=(),
    known_entries=None,
) -> MeetingContextPack:
    if any(isinstance(v, str) for v in (agenda, terms, participants, documents)):
        raise InvalidContext("Context collections must be arrays, not strings.")
    agenda, terms, participants, documents = map(tuple, (agenda, terms, participants, documents))
    if len(documents) > 8 or len(terms) > 256:
        raise InvalidContext("Too many context documents or terms.")
    known = known_entries if known_entries is not None else load_glossary().entries
    sources, explicit, extracted = [], {}, {}

    def source(kind, label, payload):
        item = ContextSource(f"ctx_{len(sources) + 1:04d}", kind, label, sha256(payload))
        sources.append(item)
        return item.id

    def add(value, sid, *, inferred=False):
        data = {"canonical": value} if isinstance(value, str) else value
        if not isinstance(data, dict) or set(data) - {
            "canonical",
            "aliases",
            "category",
            "scope",
            "priority",
        }:
            raise InvalidContext("Terms must be names or supported term objects.")
        if "canonical" not in data or not isinstance(data.get("aliases", ()), (list, tuple)):
            raise InvalidContext("Terms require a canonical name and alias array.")
        text(data["canonical"])
        scope = data.get("scope", "meeting")
        if scope not in ("meeting", "project"):
            raise InvalidContext("Context scope must be meeting or project.")
        key = (scope, normalize(data["canonical"]))
        identity = "context." + sha256(scope + ":" + key[1])[:20]
        term = ContextTerm(
            identity,
            data["canonical"],
            tuple(data.get("aliases", ())),
            data.get("category", "technology"),
            scope,
            (sid,),
            data.get("priority", 0.6 if inferred else 0.9),
        )
        target = explicit if not inferred else extracted
        previous = explicit.get(key) or extracted.get(key)
        if previous:
            preferred = term if not inferred and key not in explicit else previous
            merged = replace(
                preferred,
                aliases=tuple(dict.fromkeys(previous.aliases + term.aliases)),
                source_ids=tuple(dict.fromkeys(previous.source_ids + term.source_ids)),
                priority=max(previous.priority, term.priority),
            )
            if not inferred:
                extracted.pop(key, None)
                explicit[key] = merged
            else:
                (explicit if key in explicit else target)[key] = merged
        else:
            target[key] = term
        if len(explicit) + len(extracted) > 256:
            raise InvalidContext("Extracted context exceeds the term budget.")

    if terms:
        sid = source("glossary", "Explicit vocabulary", json.dumps(terms, sort_keys=True))
        for value in terms:
            add(value, sid)
    if participants:
        sid = source("participants", "Supplied participant names", json.dumps(participants))
        for name in participants:
            add({"canonical": name, "category": "participant_name"}, sid)
    for kind, values in (
        ("title", (title,) if title else ()),
        ("agenda", agenda),
        ("description", (description,) if description else ()),
    ):
        for i, content in enumerate(values):
            if not isinstance(content, str) or len(content) > 8000:
                raise InvalidContext("Context descriptions must be bounded text.")
            sid = source(kind, f"{kind} {i + 1}", content)
            for value in extract_terms(content, known):
                add(value, sid, inferred=True)
    total_document_bytes = 0
    for supplied in documents:
        path = Path(supplied)
        if path.suffix.lower() not in (".txt", ".md", ".json", ".csv") or path.is_symlink():
            raise InvalidContext("Context documents must be local TXT/MD/JSON/CSV files.")
        try:
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
                raise InvalidContext("Context documents must be regular files, without links.")
            with path.open("rb") as stream:
                raw = stream.read(65537)
            if len(raw) > 65536:
                raise InvalidContext("Context document exceeds 64 KiB.")
            total_document_bytes += len(raw)
            if total_document_bytes > 131072:
                raise InvalidContext("Context documents exceed the total 128 KiB budget.")
            content = raw.decode("utf-8-sig")
        except (OSError, UnicodeError) as exc:
            raise InvalidContext("Cannot read UTF-8 context document.") from exc
        sid = source("document", path.name, raw)
        try:
            if path.suffix.lower() == ".json":
                data = json.loads(content)
                if not isinstance(data, dict) or set(data) - {"terms", "text"}:
                    raise InvalidContext("Document JSON supports terms and text only.")
                if not isinstance(data.get("terms", []), list) or not isinstance(
                    data.get("text", ""), str
                ):
                    raise InvalidContext("Document JSON has invalid terms/text.")
                for value in data.get("terms", []):
                    add(value, sid)
                content = data.get("text", "")
            elif path.suffix.lower() == ".csv":
                reader = csv.DictReader(io.StringIO(content))
                if not reader.fieldnames or set(reader.fieldnames) - {
                    "canonical",
                    "aliases",
                    "category",
                    "scope",
                }:
                    raise InvalidContext("CSV needs canonical and optional aliases/category/scope.")
                for row in reader:
                    value = {key: item for key, item in row.items() if item and key != "aliases"}
                    value["aliases"] = [
                        a.strip() for a in row.get("aliases", "").split("|") if a.strip()
                    ]
                    add(value, sid)
                content = ""
        except (ValueError, TypeError, csv.Error) as exc:
            raise InvalidContext("Malformed structured context document.") from exc
        for value in extract_terms(content, known):
            add(value, sid, inferred=True)
    return MeetingContextPack(
        title,
        agenda,
        description,
        tuple(dict.fromkeys(participants)),
        tuple(explicit.values()),
        tuple(extracted.values()),
        tuple(sources),
    )


def context_entries(pack):
    """Use Phase IV's existing meeting/project scope mechanism without changing scores."""
    return tuple(
        GlossaryEntry(
            t.id,
            t.canonical,
            t.category,
            "meeting",
            "Supplied contextual vocabulary: " + t.canonical,
            aliases=t.aliases,
            source=t.scope,
            priority=t.priority,
        )
        for t in pack.terms
    )
