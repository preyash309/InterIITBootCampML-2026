"""Only timestamp-local exact known candidates may be linked; no arbitrary vocabulary."""

from meeting_assistant.grounding.normalization import normalize
from meeting_assistant.refinement.validation import protected_signature

from .models import CandidateLink


def window_pass1_text(source, window):
    selected = []
    for utterance in source.utterances:
        if utterance.words:
            selected.extend(
                w.text
                for w in utterance.words
                if window.start <= (w.start + w.end) / 2 < window.end
            )
        elif utterance.start >= window.start and utterance.end <= window.end:
            selected.append(utterance.text)
    return " ".join(value.strip() for value in selected)


def link_candidates(window, grounding, segments, terms, pass1_text, pass2_text):
    # Optional second-pass audio includes protected content. Disagreement vetoes all links.
    conflicts = (
        ()
        if protected_signature(pass1_text) == protected_signature(pass2_text)
        else ("protected_window_disagreement",)
    )
    if conflicts:
        return (), conflicts
    words = tuple(word for segment in segments for word in segment.words)
    allowed = {term.entry_id for term in terms}
    links = []
    for record in grounding.records:
        if record.id not in window.grounding_ids:
            continue
        for candidate in record.candidates:
            if candidate.entry_id not in allowed:
                continue
            for i in range(len(words)):
                for count in range(1, min(6, len(words) - i) + 1):
                    group = words[i : i + count]
                    if normalize(" ".join(w.text.strip() for w in group)) != normalize(
                        candidate.canonical
                    ):
                        continue
                    if group[0].start <= record.end + 0.75 and group[-1].end >= record.start - 0.75:
                        links.append(
                            CandidateLink(
                                record.id, candidate.entry_id, group[0].start, group[-1].end
                            )
                        )
                        break
                if any(
                    link.grounding_id == record.id and link.candidate_entry_id == candidate.entry_id
                    for link in links
                ):
                    break
    return tuple(links), ()
