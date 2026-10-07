"""Build packaged JSONL from original, reviewed domain seed lists and overrides.

No scraping, invented combinations, or plural expansion. Repeated concepts across
subjects are retained once with their first domain. IDs are content-stable.
Edit glossary_seed.txt/overrides.json then rebuild; --check detects stale output.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from meeting_assistant.grounding.glossary import build_glossary
from meeting_assistant.grounding.models import GlossaryEntry
from meeting_assistant.grounding.normalization import normalize

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    overrides = json.loads((ROOT / "scripts/glossary_overrides.json").read_text(encoding="utf-8"))
    entries, seen, repeated = [], set(), []
    domain = category = description = None
    for line in (ROOT / "scripts/glossary_seed.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        if line.startswith("@"):
            domain, category, description = line[1:].split("|", 2)
            continue
        for canonical in line.split(";"):
            canonical = canonical.strip()
            key = normalize(canonical)
            if key in seen:
                repeated.append(canonical)
                continue
            seen.add(key)
            data = {
                "id": "global." + hashlib.sha256(key.encode()).hexdigest()[:16],
                "canonical": canonical,
                "category": category,
                "domain": domain,
                "description": f"{canonical}: {description}",
            }
            data.update(overrides.get(canonical, {}))
            for name in ("aliases", "asr_aliases", "ambiguous_aliases", "symbols"):
                data[name] = tuple(data.get(name, ()))
            entries.append(GlossaryEntry(**data))
    unknown = set(overrides) - {e.canonical for e in entries}
    if unknown:
        raise ValueError(f"Unknown overrides: {sorted(unknown)}")
    glossary = build_glossary(entries)
    by_domain = defaultdict(list)
    from dataclasses import asdict

    for entry in glossary.entries:
        by_domain[entry.domain].append(
            json.dumps(asdict(entry), ensure_ascii=False, sort_keys=True)
        )
    root = ROOT / "src/meeting_assistant/grounding/data/glossary"
    root.mkdir(parents=True, exist_ok=True)
    for domain, lines in by_domain.items():
        text = "\n".join(lines) + "\n"
        path = root / (domain + ".jsonl")
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Stale glossary output: {path.name}")
        else:
            path.write_text(text, encoding="utf-8", newline="\n")
    print(json.dumps(glossary.stats(), indent=2))
    print(f"Repeated cross-domain concepts retained once: {len(repeated)}")


if __name__ == "__main__":
    main()
