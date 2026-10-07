import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from meeting_assistant.asr.models import TranscriptSegment
from meeting_assistant.diarization.reconciliation import reconcile_transcript
from meeting_assistant.diarization.serialization import speaker_transcript_to_json
from meeting_assistant.grounding.config import GroundingConfig
from meeting_assistant.grounding.embeddings import index_embeddings
from meeting_assistant.grounding.exceptions import EmbeddingError, InvalidGroundingResult
from meeting_assistant.grounding.glossary import build_glossary
from meeting_assistant.grounding.normalization import phonetic
from meeting_assistant.grounding.retrieval import GroundingRetriever
from meeting_assistant.grounding.service import ground_span, ground_transcript
from tests.diarization.helpers import diar, raw

from .helpers import StubEmbedding, entry, speaker


class RetrievalTests(unittest.TestCase):
    def test_empty_transcript_has_no_records(self):
        result = ground_transcript(reconcile_transcript(raw(), diar()), retriever=self.engine)
        self.assertEqual(result.records, ())
        self.assertEqual(result.processing_info.considered_spans, 0)

    def test_segment_only_fallback_preserves_coarse_timing(self):
        source = reconcile_transcript(
            raw(segments=(TranscriptSegment("seg_000001", 0, 5, "cue drant", ()),)), diar()
        )
        result = ground_transcript(source, retriever=self.engine)
        record = result.records[0]
        self.assertEqual(record.source_word_references, ())
        self.assertEqual(record.source_segment_ids, ("seg_000001",))
        self.assertEqual((record.start, record.end), (0, 5))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = GroundingConfig(index_cache=Path(self.temp.name))
        self.backend = StubEmbedding()
        self.engine = GroundingRetriever(
            build_glossary([entry()]), self.config, embeddings=self.backend
        )

    def test_exact_canonical(self):
        result = self.engine.candidates("Qdrant", "Vector search")
        self.assertEqual(result[0].match_type, "canonical_exact")

    def test_registered_alias_spacing(self):
        engine = GroundingRetriever(
            build_glossary([entry(canonical="Kubernetes", asr_aliases=("cube net ease",))]),
            self.config,
            embeddings=self.backend,
        )
        c = engine.candidates("CubeNet Ease", "containers and pods")[0]
        self.assertEqual(c.canonical, "Kubernetes")
        self.assertEqual(c.match_type, "normalized_exact")
        self.assertIn("registered ASR alias", c.reasons)

    def test_unspoken_prefix_is_not_inferred(self):
        engine = GroundingRetriever(
            build_glossary([entry(canonical="pgvector", asr_aliases=())]),
            self.config,
            embeddings=self.backend,
        )
        self.assertEqual(
            engine.candidates("vector", "vector database", semantic_scores=np.array([0.9])), ()
        )

    def test_fuzzy_requires_context_evidence(self):
        self.assertEqual(
            self.engine.candidates("qdrantt", "unrelated", semantic_scores=np.array([0.01])), ()
        )

    def test_explicit_asr_alias(self):
        result = self.engine.candidates("cue drant", "Vector search")
        self.assertEqual(result[0].canonical, "Qdrant")
        self.assertEqual(result[0].match_type, "asr_alias_exact")

    def test_normalized_exact(self):
        self.assertEqual(
            self.engine.candidates("QDRANT", "Vector search")[0].match_type, "normalized_exact"
        )

    def test_lexical_fuzzy(self):
        c = self.engine.candidates("qdrantt", "Vector search")[0]
        self.assertEqual(c.match_type, "fuzzy")
        self.assertGreater(c.lexical_score, 0.8)

    def test_phonetic_features_offline(self):
        self.assertEqual(phonetic("pie torch"), phonetic("PyTorch"))
        self.assertTrue(phonetic("Kubernetes"))

    def test_component_scores_bounded(self):
        c = self.engine.candidates("cue drant", "Vector search")[0]
        for score in (c.score, c.lexical_score, c.phonetic_score, c.semantic_score, c.scope_score):
            self.assertTrue(0 <= score <= 1)

    def test_scope_priority_for_comparable_matches(self):
        engine = GroundingRetriever(
            build_glossary([entry(), entry(id="meeting.qdrant", source="meeting")]),
            self.config,
            embeddings=self.backend,
        )
        self.assertEqual(engine.candidates("Qdrant", "Vector search")[0].scope, "meeting")

    def test_context_gate_common_word(self):
        engine = GroundingRetriever(
            build_glossary([entry(canonical="React", asr_aliases=(), requires_context=True)]),
            self.config,
            embeddings=self.backend,
        )
        self.assertEqual(
            engine.candidates("React", "calm response", semantic_scores=np.array([0.05])), ()
        )
        self.assertTrue(
            engine.candidates("React", "UI components", semantic_scores=np.array([0.8]))
        )

    def test_protected_numbers_and_negations(self):
        for text in ("15 percent", "not approved", "Friday", "three", "thanks"):
            self.assertEqual(self.engine.candidates(text, "Context"), ())

    def test_model_digits_require_exact_variant(self):
        engine = GroundingRetriever(
            build_glossary([entry(canonical="YOLOv8", asr_aliases=())]),
            self.config,
            embeddings=self.backend,
        )
        self.assertTrue(engine.candidates("YOLOv8", "object detection"))
        self.assertEqual(engine.candidates("YOLOv9", "object detection"), ())

    def test_no_shortlist_skips_embeddings(self):
        self.assertEqual(self.engine.candidates("zzzzzzzzzzz", "Context"), ())
        self.assertEqual(self.backend.calls, [])

    def test_backend_reused(self):
        ground_span("cue drant", "Vector search", retriever=self.engine)
        ground_span("cue drant", "Vector search", retriever=self.engine)
        self.assertEqual(
            sum(len(call) for call in self.backend.calls if call[0].startswith("Qdrant.")), 1
        )

    def test_source_immutable_and_refs_exact(self):
        source = speaker()
        before = speaker_transcript_to_json(source)
        result = ground_transcript(source, retriever=self.engine)
        self.assertEqual(before, speaker_transcript_to_json(source))
        self.assertEqual(result.source_speaker_sha256, hashlib.sha256(before.encode()).hexdigest())
        r = next(r for r in result.records if r.observed_text == "cue drant")
        self.assertEqual([ref.word_index for ref in r.source_word_references], [1, 2])
        self.assertEqual(source.utterances[0].text[r.char_start : r.char_end], r.observed_text)
        self.assertEqual(r.speaker_id, "SPEAKER_00")

    def test_records_nonoverlapping_deterministic(self):
        source = speaker()
        a = ground_transcript(source, retriever=self.engine)
        b = ground_transcript(source, retriever=self.engine)
        self.assertEqual(a.records, b.records)
        for first, second in zip(a.records, a.records[1:]):
            self.assertLessEqual(first.char_end, second.char_start)

    def test_context_bounded(self):
        source = speaker("cue drant " * 40)
        engine = GroundingRetriever(
            self.engine.glossary, replace(self.config, context_chars=50), embeddings=self.backend
        )
        result = ground_transcript(source, retriever=engine)
        self.assertTrue(all(len(r.context) <= 50 for r in result.records))

    def test_invalid_source(self):
        with self.assertRaises(InvalidGroundingResult):
            ground_transcript({}, retriever=self.engine)


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = GroundingConfig(index_cache=Path(self.temp.name))
        self.backend = StubEmbedding()
        self.glossary = build_glossary([entry()])

    def test_cache_hit_without_reencoding(self):
        first, hit = index_embeddings(self.glossary, self.backend, self.config)
        second, hit = index_embeddings(self.glossary, self.backend, self.config)
        self.assertTrue(hit)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(self.backend.calls), 1)

    def test_changed_entry_only_encoded(self):
        index_embeddings(self.glossary, self.backend, self.config)
        updated = build_glossary(
            [entry(), entry(canonical="Other", id="global.other", asr_aliases=())]
        )
        index_embeddings(updated, self.backend, self.config)
        self.assertEqual(len(self.backend.calls[-1]), 1)
        self.assertTrue(self.backend.calls[-1][0].startswith("Other."))

    def test_description_invalidates_row(self):
        index_embeddings(self.glossary, self.backend, self.config)
        index_embeddings(
            build_glossary([entry(description="New description.")]), self.backend, self.config
        )
        self.assertEqual(len(self.backend.calls), 2)

    def test_meeting_embeddings_not_persisted(self):
        glossary = build_glossary(
            [
                entry(),
                entry(
                    id="meeting.private",
                    canonical="Private Person",
                    source="meeting",
                    asr_aliases=(),
                ),
            ]
        )
        index_embeddings(glossary, self.backend, self.config)
        files = list(Path(self.temp.name).rglob("manifest.json"))
        self.assertEqual(len(files), 1)
        self.assertNotIn("Private Person", files[0].read_text())
        self.assertEqual(len(np.load(files[0].parent / "vectors.npy", allow_pickle=False)), 1)

    def test_corrupt_cache_fails_explicitly(self):
        index_embeddings(self.glossary, self.backend, self.config)
        next(Path(self.temp.name).rglob("vectors.npy")).write_bytes(b"broken")
        with self.assertRaises(EmbeddingError):
            index_embeddings(self.glossary, self.backend, self.config)

    def test_malformed_embedding_shape(self):
        self.backend.encode = lambda texts: np.zeros((len(texts), 3), dtype=np.float32)
        with self.assertRaises(EmbeddingError):
            index_embeddings(self.glossary, self.backend, self.config)
