"""Alias-first bounded lexical/phonetic retrieval with context cosine evidence."""

import threading
import time

from .config import GroundingConfig
from .embeddings import EmbeddingBackend, MiniLMBackend, index_embeddings
from .exceptions import EmbeddingError
from .glossary import Glossary
from .models import GroundingCandidate
from .normalization import normalize, phonetic, protected

STOP = frozenset(
    "a an the and or but if then this that these those it its is are was were be been being we our us you your he she they them their my i me to of for in on at as by with from please yes no okay thanks let can could would may might do does did have has had use using need want get make take go come said say also about more some any all so very much here there which what when where how why one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion monday tuesday wednesday thursday friday saturday sunday january february march april june july august september october november december".split()
)


class GroundingRetriever:
    def __init__(
        self,
        glossary: Glossary,
        config: GroundingConfig | None = None,
        *,
        embeddings: EmbeddingBackend | None = None,
    ):
        self.glossary = glossary
        self.config = config or GroundingConfig()
        self.embeddings = embeddings or MiniLMBackend(self.config)
        self._matrix = None
        self._lock = threading.Lock()
        self._variants = []
        self._entry_indices = []
        self._kinds = []
        for index, entry in enumerate(glossary.entries):
            for kind, variants in (
                ("canonical_exact", (entry.canonical,)),
                ("alias_exact", entry.aliases),
                ("asr_alias_exact", entry.asr_aliases),
            ):
                for value in variants:
                    self._variants.append(value)
                    self._entry_indices.append(index)
                    self._kinds.append(kind)
        self._normalized = [normalize(v) for v in self._variants]
        self._phonetics = [phonetic(v) for v in self._variants]
        self.cache_hit = False
        self.index_load_seconds = 0.0
        self.model_load_seconds = 0.0

    def prepare(self) -> None:
        with self._lock:
            if self._matrix is None:
                if hasattr(self.embeddings, "load_model"):
                    self.model_load_seconds = self.embeddings.load_model()
                started = time.perf_counter()
                self._matrix, self.cache_hit = index_embeddings(
                    self.glossary, self.embeddings, self.config
                )
                self.index_load_seconds = time.perf_counter() - started

    def context_scores(self, contexts: list[str]):
        import numpy as np

        self.prepare()
        vectors = self.embeddings.encode([c[: self.config.context_chars] for c in contexts])
        vectors = np.asarray(vectors, dtype=np.float32)
        if (
            vectors.shape != (len(contexts), self.embeddings.dimension)
            or not np.isfinite(vectors).all()
        ):
            raise EmbeddingError("Malformed context embedding output.")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if (norms < 1e-8).any():
            raise EmbeddingError("Empty context embedding output.")
        return np.clip((vectors / norms) @ self._matrix.T, 0, 1)

    def candidates(
        self, span: str, context: str, *, semantic_scores=None
    ) -> tuple[GroundingCandidate, ...]:
        import numpy as np
        from rapidfuzz import fuzz, process

        normalized = normalize(span)
        if not normalized or protected(span) or normalized in STOP:
            return ()
        digits = any(c.isdigit() for c in normalized)
        # Native ratios avoid O(Python words * glossary size) token loops.
        lexical = (
            process.cdist([normalized], self._normalized, scorer=fuzz.ratio, dtype=np.float32)[0]
            / 100
        )
        token = (
            process.cdist(
                [normalized], self._normalized, scorer=fuzz.token_sort_ratio, dtype=np.float32
            )[0]
            / 100
        )
        lexical = 0.75 * lexical + 0.25 * token
        code = phonetic(span)
        phone = (
            (process.cdist([code], self._phonetics, scorer=fuzz.ratio, dtype=np.float32)[0] / 100)
            if code
            else np.zeros(len(lexical))
        )
        exact = [
            i
            for i, value in enumerate(self._normalized)
            if value.replace(" ", "") == normalized.replace(" ", "")
        ]
        if digits or len(normalized.replace(" ", "")) <= 2:
            pool = exact
        else:
            pool = np.flatnonzero(
                (lexical >= self.config.lexical_floor) | (phone >= self.config.phonetic_floor)
            ).tolist()
        if not pool:
            return ()
        if semantic_scores is None:
            semantic_scores = self.context_scores([context])[0]
        candidates = {}
        for variant_index in pool:
            entry_index = self._entry_indices[variant_index]
            entry = self.glossary.entries[entry_index]
            semantic = float(semantic_scores[entry_index])
            is_exact = variant_index in exact
            # Substrings cannot justify adding an unspoken qualifier or product prefix.
            if not is_exact and normalized.replace(" ", "") in self._normalized[
                variant_index
            ].replace(" ", ""):
                continue
            if not is_exact and semantic < self.config.context_floor:
                continue
            if entry.requires_context and semantic < self.config.context_floor:
                continue
            if digits and not is_exact:
                continue
            scope = {"global": 0.2, "project": 0.6, "meeting": 1.0}[entry.source]
            scope = 0.8 * scope + 0.2 * entry.priority
            lex, pho = float(lexical[variant_index]), float(phone[variant_index])
            kind = (
                self._kinds[variant_index]
                if span == self._variants[variant_index]
                else ("normalized_exact" if is_exact else "fuzzy")
            )
            score = (
                self.config.lexical_weight * lex
                + self.config.phonetic_weight * pho
                + self.config.semantic_weight * semantic
                + self.config.scope_weight * scope
            )
            # Registered exact variants dominate fuzzy evidence without pretending to be calibrated.
            if is_exact:
                score = max(score, 0.90 + 0.05 * semantic + 0.05 * scope)
            if not is_exact and score < self.config.min_score:
                continue
            reasons = [
                "normalized registered variant" if is_exact else "lexical/phonetic shortlist",
                "bounded-context cosine",
            ]
            if entry.requires_context:
                reasons.append("ambiguous term passed context gate")
            if is_exact and self._kinds[variant_index] == "asr_alias_exact":
                reasons.append("registered ASR alias")
            candidate = GroundingCandidate(
                entry.id,
                entry.canonical,
                entry.source,
                entry.domain,
                entry.category,
                entry.description,
                self._variants[variant_index],
                kind,
                min(score, 1),
                lex,
                pho,
                semantic,
                scope,
                tuple(reasons),
            )
            previous = candidates.get(entry.id)
            if previous is None or candidate.score > previous.score:
                candidates[entry.id] = candidate
        return tuple(
            sorted(candidates.values(), key=lambda c: (-c.score, -c.scope_score, c.entry_id))[
                : self.config.top_k
            ]
        )
