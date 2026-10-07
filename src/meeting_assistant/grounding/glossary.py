"""Versioned, validated glossary layers. Meeting-specific data stays in memory."""

import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .exceptions import InvalidGlossary
from .models import GlossaryEntry
from .normalization import normalize

DATA = Path(__file__).parent / "data" / "glossary"


@dataclass(frozen=True)
class Glossary:
    entries: tuple[GlossaryEntry, ...]
    version: str

    def stats(self) -> dict:
        return {
            "canonical_entries": len(self.entries),
            "aliases": sum(len(e.aliases) for e in self.entries),
            "asr_aliases": sum(len(e.asr_aliases) for e in self.entries),
            "descriptions": sum(bool(e.description) for e in self.entries),
            "ambiguity_marked_entries": sum(
                bool(e.ambiguous_aliases or e.sense or e.requires_context) for e in self.entries
            ),
            "domains": dict(sorted(Counter(e.domain for e in self.entries).items())),
            "categories": dict(sorted(Counter(e.category for e in self.entries).items())),
            "scopes": dict(sorted(Counter(e.source for e in self.entries).items())),
            "version": self.version,
        }


def entry_from_dict(data: dict) -> GlossaryEntry:
    try:
        if not isinstance(data, dict) or any(
            not isinstance(data.get(name, []), (list, tuple))
            for name in ("aliases", "asr_aliases", "ambiguous_aliases", "symbols")
        ):
            raise InvalidGlossary("Aliases must be arrays, not strings.")
        return GlossaryEntry(
            **{
                **data,
                **{
                    name: tuple(data.get(name, ()))
                    for name in ("aliases", "asr_aliases", "ambiguous_aliases", "symbols")
                },
            }
        )
    except (TypeError, AttributeError, ValueError) as exc:
        raise InvalidGlossary("Malformed glossary entry.") from exc


def build_glossary(entries: Iterable[GlossaryEntry]) -> Glossary:
    supplied = tuple(entries)
    if not supplied or any(not isinstance(e, GlossaryEntry) for e in supplied):
        raise InvalidGlossary("Glossary needs nonempty typed entries.")
    ordered = tuple(sorted(supplied, key=lambda e: e.id))
    ids, canonical = set(), defaultdict(list)
    aliases = defaultdict(list)
    for entry in ordered:
        if not isinstance(entry, GlossaryEntry) or entry.id in ids:
            raise InvalidGlossary("Glossary IDs must be unique typed entries.")
        ids.add(entry.id)
        if re.fullmatch(r"[a-z][a-z0-9_]{0,80}", entry.category) is None:
            raise InvalidGlossary("Categories must be lowercase stable identifiers.")
        canonical[(entry.source, normalize(entry.canonical))].append(entry)
        for value in (entry.canonical, *entry.aliases, *entry.asr_aliases):
            if not normalize(value):
                raise InvalidGlossary("Normalized glossary variants cannot be empty.")
            aliases[(entry.source, normalize(value).replace(" ", ""))].append(entry)
    for group in canonical.values():
        if len(group) > 1 and (
            any(not e.sense for e in group) or len({e.sense for e in group}) != len(group)
        ):
            raise InvalidGlossary("Duplicate canonical names require distinct explicit senses.")
    for (_, value), group in aliases.items():
        unique = {e.id: e for e in group}
        if len(unique) > 1 and any(
            value not in {normalize(a).replace(" ", "") for a in e.ambiguous_aliases}
            for e in unique.values()
        ):
            raise InvalidGlossary("Alias collisions require ambiguity metadata on every entry.")
    payload = json.dumps(
        [asdict(e) for e in ordered], sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return Glossary(ordered, hashlib.sha256(payload.encode("utf-8")).hexdigest())


def read_entries(
    path: Path | str, *, expected_scope: str | None = None
) -> tuple[GlossaryEntry, ...]:
    entries = []
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                if len(line) > 32_000 or len(entries) >= 20_000:
                    raise InvalidGlossary("Glossary exceeds bounded entry/line limits.")
                if line.strip():
                    entry = entry_from_dict(json.loads(line))
                    if expected_scope and entry.source != expected_scope:
                        raise InvalidGlossary("Glossary entry belongs to the wrong scope.")
                    entries.append(entry)
    except (OSError, ValueError, TypeError) as exc:
        raise InvalidGlossary("Cannot read glossary JSONL; check its format and access.") from exc
    return tuple(entries)


def load_glossary(
    *, project_glossary: Path | None = None, meeting_entries: Iterable[GlossaryEntry] = ()
) -> Glossary:
    entries = []
    for path in sorted(DATA.glob("*.jsonl")):
        entries.extend(read_entries(path, expected_scope="global"))
    if not entries:
        raise InvalidGlossary("Packaged global glossary is missing.")
    if project_glossary:
        entries.extend(read_entries(project_glossary, expected_scope="project"))
    for entry in meeting_entries:
        if entry.source != "meeting":
            raise InvalidGlossary("Dynamic glossary entries must use meeting scope.")
        entries.append(entry)
    return build_glossary(entries)


def read_meeting_context(path: Path | str) -> tuple[GlossaryEntry, ...]:
    """Bounded JSON: custom entries plus participant/company/project name lists."""
    try:
        with Path(path).open(encoding="utf-8-sig") as stream:
            payload = stream.read(262_145)
        if len(payload) > 262_144:
            raise InvalidGlossary("Meeting context is too large.")
        data = json.loads(payload)
        if not isinstance(data, dict) or set(data) - {
            "entries",
            "participants",
            "companies",
            "projects",
        }:
            raise InvalidGlossary("Unknown meeting context fields.")
        entries = [entry_from_dict(e) for e in data.get("entries", [])]
        for kind in ("participants", "companies", "projects"):
            names = data.get(kind, [])
            if not isinstance(names, list) or any(
                not isinstance(n, str) or not n.strip() or len(n) > 120 for n in names
            ):
                raise InvalidGlossary("Context names must be bounded nonempty strings.")
            for name in names:
                identity = hashlib.sha256((kind + name).encode()).hexdigest()[:20]
                entries.append(
                    GlossaryEntry(
                        "meeting." + identity,
                        name,
                        kind,
                        "meeting",
                        f"{name}: supplied meeting { {'participants': 'participant', 'companies': 'company', 'projects': 'project'}[kind] } name.",
                        source="meeting",
                        priority=1,
                    )
                )
        if len(entries) > 1000 or any(e.source != "meeting" for e in entries):
            raise InvalidGlossary(
                "Meeting context must contain at most 1000 meeting-scoped entries."
            )
        return tuple(entries)
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        raise InvalidGlossary("Malformed meeting context JSON.") from exc
