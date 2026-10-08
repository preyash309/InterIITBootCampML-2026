"""Create a context pack without provider calls or document-processing dependencies."""

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from .context import build_meeting_context
from .exceptions import ContextualASRError
from .serialization import context_to_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build an auditable meeting context pack; no model calls."
    )
    parser.add_argument("--title")
    parser.add_argument("--agenda", action="append", default=[])
    parser.add_argument("--description")
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--participant", action="append", default=[])
    parser.add_argument("--document", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    temporary = None
    try:
        context = build_meeting_context(
            title=args.title,
            agenda=args.agenda,
            description=args.description,
            terms=args.term,
            participants=args.participant,
            documents=args.document,
        )
        if args.output.exists():
            raise ContextualASRError("Context output exists; choose a new path.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.parent / (".context-" + uuid4().hex + ".tmp")
        with temporary.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(context_to_json(context))
            stream.flush()
            os.fsync(stream.fileno())
        os.rename(temporary, args.output)
        print(
            f"Context pack: {args.output}; {len(context.terms)} terms; {len(context.sources)} sources"
        )
        return 0
    except (ContextualASRError, OSError) as exc:
        print(
            f"Context creation failed [{getattr(exc, 'code', 'context_write_failed')}]: check context inputs/output access.",
            file=sys.stderr,
        )
        return 1
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
