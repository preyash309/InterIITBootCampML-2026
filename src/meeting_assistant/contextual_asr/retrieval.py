"""Small per-window vocabulary; never send the complete glossary or document prose."""

from meeting_assistant.grounding.normalization import normalize
from meeting_assistant.refinement.validation import protected_signature

from .models import RetrievedTerm


def retrieve_terms(window, grounding, context, config):
    source_map = {t.id: t.source_ids for t in context.terms} if context else {}
    ranked = []
    for record in grounding.records:
        if record.id not in window.grounding_ids:
            continue
        for candidate in record.candidates:
            if protected_signature(candidate.canonical):
                continue
            if candidate.entry_id not in source_map and not config.use_global_glossary:
                continue
            ranked.append(
                (
                    candidate.entry_id in source_map,
                    candidate.scope_score,
                    candidate.score,
                    candidate,
                )
            )
    seen, selected = set(), []
    for _, _, _, candidate in sorted(ranked, key=lambda v: (-v[0], -v[1], -v[2], v[3].entry_id)):
        key = normalize(candidate.canonical)
        if key not in seen:
            seen.add(key)
            selected.append(
                RetrievedTerm(
                    candidate.entry_id,
                    candidate.canonical,
                    candidate.scope,
                    source_map.get(candidate.entry_id, ()),
                )
            )
        if len(selected) == config.top_k_terms:
            break
    return tuple(selected)


def build_prompt(terms):
    # UTF-8 bytes are a conservative token upper bound for Whisper's byte BPE.
    # Stay below Groq's documented 224-token limit without a tokenizer dependency.
    prefix = "Possible vocabulary: "
    names, retained = [], []
    for term in terms:
        candidate = prefix + ", ".join([*names, term.canonical]) + "."
        if len(candidate.encode("utf-8")) > 223:
            continue
        names.append(term.canonical)
        retained.append(term)
    return (prefix + ", ".join(names) + "." if names else ""), tuple(retained)
