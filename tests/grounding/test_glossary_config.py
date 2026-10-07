import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

from meeting_assistant.grounding.config import GroundingConfig
from meeting_assistant.grounding.exceptions import GroundingConfigurationError, InvalidGlossary
from meeting_assistant.grounding.glossary import (
    build_glossary,
    entry_from_dict,
    load_glossary,
    read_entries,
    read_meeting_context,
)
from meeting_assistant.grounding.models import GlossaryEntry
from meeting_assistant.grounding.normalization import normalize, protected


def item(name="Qdrant", **kw):
    return GlossaryEntry(
        **{
            **dict(
                id=name.lower(),
                canonical=name,
                domain="retrieval",
                category="search",
                description="Vector search database.",
            ),
            **kw,
        }
    )


class GlossaryTests(unittest.TestCase):
    def test_units_have_symbols_without_automatic_aliases(self):
        entries = {e.canonical: e for e in load_glossary().entries}
        self.assertEqual(entries["Celsius"].symbols, ("°C",))
        self.assertNotIn("°C", entries["Celsius"].aliases)
        self.assertEqual(sum(e.category == "engineering_units" for e in entries.values()), 20)

    def test_global_curated_count_domains(self):
        stats = load_glossary().stats()
        self.assertEqual(stats["canonical_entries"], 3590)
        self.assertEqual(len(stats["domains"]), 17)
        self.assertEqual(stats["descriptions"], 3590)

    def test_ids_unique(self):
        with self.assertRaises(InvalidGlossary):
            build_glossary([item(), item("Other", id="qdrant")])

    def test_duplicate_canonical_requires_sense(self):
        with self.assertRaises(InvalidGlossary):
            build_glossary([item(), item(id="other")])

    def test_explicit_senses_allowed(self):
        values = [
            item(id="one", sense="database", ambiguous_aliases=("Qdrant",)),
            item(id="two", sense="product", ambiguous_aliases=("Qdrant",)),
        ]
        self.assertEqual(len(build_glossary(values).entries), 2)

    def test_alias_collision_rejected(self):
        with self.assertRaises(InvalidGlossary):
            build_glossary([item(aliases=("same",)), item("Other", aliases=("same",))])

    def test_marked_collision_preserved(self):
        glossary = build_glossary(
            [
                item(aliases=("same",), ambiguous_aliases=("same",)),
                item("Other", aliases=("same",), ambiguous_aliases=("same",)),
            ]
        )
        self.assertEqual(len(glossary.entries), 2)

    def test_cross_scope_preserves_provenance(self):
        glossary = build_glossary([item(), item(id="meeting.qdrant", source="meeting")])
        self.assertEqual({e.source for e in glossary.entries}, {"global", "meeting"})

    def test_version_deterministic(self):
        a, b = item(), item("Other")
        self.assertEqual(build_glossary([a, b]).version, build_glossary([b, a]).version)

    def test_version_changes_with_description(self):
        self.assertNotEqual(
            build_glossary([item()]).version,
            build_glossary([item(description="Changed meaning.")]).version,
        )

    def test_empty_glossary_rejected(self):
        with self.assertRaises(InvalidGlossary):
            build_glossary([])

    def test_invalid_entry_fields(self):
        for kw in (
            {"canonical": ""},
            {"description": ""},
            {"source": "unknown"},
            {"domain": "invalid"},
            {"priority": float("nan")},
            {"aliases": ["bad"]},
            {"acronym_expansion": 2},
        ):
            with self.subTest(kw=kw), self.assertRaises(InvalidGlossary):
                item(**kw)

    def test_alias_string_not_array(self):
        with self.assertRaises(InvalidGlossary):
            entry_from_dict({"aliases": "word"})

    def test_invalid_category(self):
        with self.assertRaises(InvalidGlossary):
            build_glossary([item(category="Bad Category")])

    def test_jsonl_malformed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            path.write_text("{broken}")
            with self.assertRaises(InvalidGlossary):
                read_entries(path)

    def test_scope_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "entry.jsonl"
            path.write_text(
                json.dumps(
                    dict(
                        id="x",
                        canonical="Name",
                        category="custom",
                        domain="meeting",
                        description="Name supplied for project.",
                        source="project",
                    )
                )
            )
            with self.assertRaises(InvalidGlossary):
                read_entries(path, expected_scope="global")

    def test_dynamic_names_custom_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text(
                json.dumps(
                    {
                        "participants": ["Asha Rao"],
                        "companies": ["Example Co"],
                        "projects": ["Project Aurora"],
                    }
                )
            )
            entries = read_meeting_context(path)
            self.assertEqual(len(entries), 3)
            self.assertTrue(all(e.source == "meeting" for e in entries))
            self.assertEqual(entries, read_meeting_context(path))

    def test_unknown_dynamic_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "context.json"
            path.write_text('{"unknown":[]}')
            with self.assertRaises(InvalidGlossary):
                read_meeting_context(path)

    def test_wrong_dynamic_scope(self):
        with self.assertRaises(InvalidGlossary):
            load_glossary(meeting_entries=(item(),))

    def test_entries_frozen(self):
        with self.assertRaises(FrozenInstanceError):
            item().canonical = "Edited"


class ConfigurationTests(unittest.TestCase):
    def test_symbols_preserve_distinct_names(self):
        self.assertEqual(len({normalize(n) for n in ("C", "C++", "C#")}), 3)
        self.assertNotEqual(normalize("U-Net"), normalize("UNet++"))

    def test_defaults(self):
        config = GroundingConfig()
        self.assertTrue(config.offline_only)
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.max_span_words, 4)

    def test_environment_overrides(self):
        config = GroundingConfig.from_env(
            environ={
                "GROUNDING_TOP_K": "3",
                "GROUNDING_MIN_SCORE": "0.7",
                "GROUNDING_MODEL_CACHE": "models",
                "GROUNDING_OFFLINE_ONLY": "true",
            }
        )
        self.assertEqual(config.top_k, 3)
        self.assertEqual(config.min_score, 0.7)
        self.assertEqual(config.model_cache, Path("models"))

    def test_invalid_environment(self):
        for values in (
            {"GROUNDING_TOP_K": "x"},
            {"GROUNDING_OFFLINE_ONLY": "yes"},
            {"GROUNDING_MIN_SCORE": "nan"},
        ):
            with self.subTest(values=values), self.assertRaises(GroundingConfigurationError):
                GroundingConfig.from_env(environ=values)

    def test_invalid_limits_weights(self):
        for kw in (
            {"top_k": 0},
            {"max_span_words": 5},
            {"context_chars": 0},
            {"batch_size": False},
            {"lexical_weight": 0.2},
            {"model": "other"},
        ):
            with self.subTest(kw=kw), self.assertRaises(GroundingConfigurationError):
                replace(GroundingConfig(), **kw)

    def test_normalization_preserves_meaning(self):
        self.assertEqual(normalize("PyTorch / PY-TORCH"), "pytorch py torch")
        self.assertEqual(normalize("ＦＡＩＳＳ"), "faiss")
        self.assertEqual(normalize("N.C.C.L."), "nccl")
        self.assertEqual(normalize("Model-16"), "model 16")

    def test_protected_negation_commitment(self):
        for text in ("not approved", "must", "won't", "cannot", "rejected", "deadline"):
            self.assertTrue(protected(text))
        self.assertFalse(protected("notebook"))

    def test_contracted_negations_are_protected(self):
        for text in ("can't", "don’t", "shouldn't", "won’t"):
            self.assertTrue(protected(text))
