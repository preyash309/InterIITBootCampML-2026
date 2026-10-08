"""Optional untrusted evidence for the existing Phase V candidate gate."""

from dataclasses import replace

from meeting_assistant.grounding.glossary import build_glossary, load_glossary
from meeting_assistant.grounding.retrieval import GroundingRetriever
from meeting_assistant.refinement.models import ContextualRefinementEvidence

from .context import context_entries
from .service import validate_contextual_source


class ContextGroundingRetriever(GroundingRetriever):
    def __init__(self, *args, context_sources, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_sources = dict(context_sources)

    def candidates(self, *args, **kwargs):
        return tuple(
            replace(
                c,
                reasons=c.reasons
                + tuple(
                    "context_source:" + identity
                    for identity in self.context_sources.get(c.entry_id, ())
                ),
            )
            for c in super().candidates(*args, **kwargs)
        )


def context_retriever(context, baseline=None, *, project_glossary=None, config=None):
    entries = context_entries(context)
    base = baseline.glossary if baseline else load_glossary(project_glossary=project_glossary)
    # Preserve global/project entries. Meeting names use Phase IV's existing scope policy.
    additions = {entry.id: entry for entry in (*base.entries, *entries)}
    return ContextGroundingRetriever(
        build_glossary(additions.values()),
        config or (baseline.config if baseline else None),
        embeddings=baseline.embeddings if baseline else None,
        context_sources={t.id: t.source_ids for t in context.terms},
    )


def refinement_evidence(result, source, grounding):
    validate_contextual_source(result, source, grounding)
    output = []
    for h in result.hypotheses:
        for link in h.candidate_links:
            output.append(
                ContextualRefinementEvidence(
                    h.id,
                    link.grounding_id,
                    link.candidate_entry_id,
                    h.pass2_text,
                    h.window.start,
                    h.window.end,
                    tuple(dict.fromkeys(s for t in h.terms for s in t.context_source_ids)),
                )
            )
    return tuple(output)


def attach_request_evidence(requests, evidence):
    # Only candidates actually sent within Phase V's top-K can be hinted.
    return tuple(
        replace(
            request,
            contextual_evidence=tuple(
                e
                for e in evidence
                if any(
                    r.id == e.grounding_record_id
                    and any(c.entry_id == e.candidate_entry_id for c in r.candidates)
                    for r in request.records
                )
            ),
        )
        for request in requests
    )
