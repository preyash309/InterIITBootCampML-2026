"""Isolated pinned Julia runtime; one engine per worker, JSON-lines only on stdout."""

import contextlib
import importlib.metadata
import json
import socket
import sys
import time
import traceback


def main():
    def refuse_connection(*_args, **_kwargs):
        raise OSError("Network connections are disabled in the local semantic worker.")

    socket.socket.connect = refuse_connection
    socket.socket.connect_ex = refuse_connection
    provider, root, device, max_length, head_length = sys.argv[1:]
    engine = None
    load_seconds = 0
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                import torch

                if engine is None:
                    before = time.monotonic()
                    if provider == "julia":
                        from julia import load_model

                        engine = load_model(
                            root,
                            device=device,
                            max_length=int(max_length),
                            head_length=int(head_length),
                            strict_encoding=True,
                            backend="torch",
                            batch_size=4,
                            marker_only_head=False,
                        )
                    elif provider == "gliner":
                        from gliner2 import AutoExtractor

                        if device == "cuda" and not torch.cuda.is_available():
                            raise RuntimeError("CUDA requested but unavailable.")
                        torch.set_num_threads(4)
                        if device == "cuda":
                            torch.cuda.reset_peak_memory_stats()
                        engine = AutoExtractor.from_pretrained(
                            root,
                            map_location=device,
                            quantize=device == "cuda",
                            local_files_only=True,
                        )
                        engine.eval()
                    else:
                        raise ValueError("Unknown local semantic provider.")
                    load_seconds = time.monotonic() - before
                before = time.monotonic()
                result = (
                    engine.predict(
                        state=render_state(payload["state"]), questions=payload["questions"]
                    )
                    if provider == "julia"
                    else gliner_predict(engine, payload, min(1024, int(max_length)))
                )
                inference_seconds = time.monotonic() - before
            result["runtime"] = {
                "device": device,
                "load_seconds": load_seconds,
                "last_inference_seconds": inference_seconds,
                "precision": ("bf16_autocast_fp32_weights" if provider == "julia" else "fp16")
                if device == "cuda"
                else "fp32",
                "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated()
                if device == "cuda"
                else None,
                "torch": str(torch.__version__),
                "transformers": importlib.metadata.version("transformers"),
                "forward_passes": len(payload["questions"]) if provider == "gliner" else 1,
                "python_socket_connections_blocked": True,
            }
        except Exception as exc:
            # Never send source paths or raw model tracebacks across the application boundary.
            result = {"error": type(exc).__name__}
            traceback.print_exc(file=sys.stderr)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


def gliner_predict(engine, payload, max_tokens):
    """Return native softmax scores for every label, without inventing probabilities."""
    answers = {}
    state = render_state(payload["state"])
    for qid, question in payload["questions"].items():
        schema = engine.create_schema().classification(
            qid,
            question["criteria"],
            prompt=question["instructions"],
            multi_label=True,
            class_act="softmax",
            cls_threshold=0,
        )
        # Pinned SDK: None means no word truncation. Inspect exact combined tokens.
        encoded = engine.processor.collate_fn_inference(
            [(state, schema.build())], max_len=None, error_policy="raise"
        )
        if encoded.input_ids.shape[-1] > max_tokens:
            raise ValueError("GLiNER combined state/schema exceeds the lossless token budget.")
        output = engine.extract(state, schema, include_confidence=True, max_len=None)[qid]
        probabilities = {item["label"]: item["confidence"] for item in output}
        if set(probabilities) != set(question["criteria"]):
            raise ValueError("GLiNER did not return all requested labels.")
        choice = max(question["criteria"], key=probabilities.__getitem__)
        answers[qid] = {
            "type": "choice",
            "choice": choice,
            "probabilities": probabilities,
            "max_probability": probabilities[choice],
        }
    return {"answers": answers}


def render_state(state):
    """Lossless evidence text with explicit roles; omit mechanical timestamp/word-index metadata.

    Source identity/timing remains in the typed result and source hashes, never inferred by the model.
    """
    if "target_id" in state:
        target = state["target_id"]
        return "\n".join(
            ("TARGET " if u["id"] == target else "CONTEXT ")
            + f"{u['id']} ({u['speaker_id']}): {u['text']}"
            for u in state["utterances"]
        )
    if "event_a" in state:
        lines = [
            f"Earlier event A ({state['event_a']['type']}): {state['event_a']['text']}",
            f"Later event B ({state['event_b']['type']}): {state['event_b']['text']}",
        ]
        lines += [
            f"Context {u['id']} ({u['speaker']}): {u['text']}"
            for u in state["conversation_between"]
        ]
        return "\n".join(lines)
    if "claim" in state:
        lines = [
            f"Claim: {state['claim']}",
            f"Record type: {state['item_type']}",
            "Existing owner: " + json.dumps(state["owner"], ensure_ascii=False),
            "Existing deadline: " + json.dumps(state["deadline_text"], ensure_ascii=False),
        ]
        lines += [
            "Evidence: "
            + json.dumps(
                {
                    k: v
                    for k, v in e.items()
                    if k in ("utterance_id", "speaker_id", "text", "refined_text", "raw_text")
                },
                ensure_ascii=False,
            )
            for e in state["evidence"]
        ]
        return "\n".join(lines)
    return json.dumps(state, ensure_ascii=False)


if __name__ == "__main__":
    raise SystemExit(main())
