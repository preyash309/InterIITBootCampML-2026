"""One serial worker, persisted queue, deliberate restart recovery; no automatic reruns."""

import logging
import traceback
from threading import Event, Thread
from time import perf_counter

from .errors import pipeline_error
from .registry import safe_path
from .schemas import STAGES

logger = logging.getLogger(__name__)


class JobWorker:
    def __init__(self, store, runner):
        self.store, self.runner = store, runner
        self.stopping = Event()
        self.wake = Event()
        self.thread = Thread(target=self._loop, name="meeting-pipeline", daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stopping.set()
        self.wake.set()
        self.thread.join()  # Current job finishes; queued jobs are interrupted on next startup.

    def _loop(self):
        while not self.stopping.is_set():
            queued = [j for j in self.store.list() if j.status == "QUEUED"]
            if queued:
                self.process(queued[-1].id)
            else:
                self.wake.wait(1)
                self.wake.clear()

    def process(self, identifier):
        job = self.store.get(identifier)
        if job.status != "QUEUED":
            return
        expected = iter(STAGES)

        def execute(stage, operation):
            if stage != next(expected, None):
                raise RuntimeError("Pipeline stages must execute exactly once in order.")
            job.status = stage.value
            job.current_stage = stage
            state = next(s for s in job.stages if s.name == stage)
            state.status = "running"
            self.store.update(job)
            started = perf_counter()
            logger.info(
                "web_stage_started meeting_id=%s stage=%s",
                identifier,
                stage.value,
                extra={"meeting_id": identifier, "stage": stage.value},
            )
            try:
                value, artifacts = operation()
                self.store.register(identifier, artifacts)
            finally:
                state.duration_seconds = perf_counter() - started
            state.status = "complete"
            self.store.update(job)
            return value

        try:
            source = safe_path(self.store.root, identifier + "/upload/source.media")
            self.runner.run(identifier, source, self.store.root, execute)
            if any(s.status != "complete" for s in job.stages):
                raise RuntimeError("Pipeline did not complete all stages.")
            required = {
                "canonical_audio",
                "raw_json",
                "speaker_json",
                "refined_json",
                "meeting_json",
                "evidence_json",
                "meeting_md",
                "raw_txt",
                "speaker_txt",
                "refined_txt",
            }
            if not required.issubset(self.store.artifacts(identifier)):
                raise RuntimeError("Pipeline did not publish required artifacts.")
            job.status = "COMPLETED"
        except Exception as exc:
            job.status = "FAILED"
            job.error = pipeline_error(exc, job.current_stage)
            for stage in job.stages:
                if stage.status == "running":
                    stage.status = "failed"
            logger.error(
                "web_pipeline_failed meeting_id=%s stage=%s code=%s exception=%s",
                identifier,
                job.current_stage,
                job.error.code,
                type(exc).__name__,
                extra={
                    "meeting_id": identifier,
                    "error_code": job.error.code,
                    "exception_type": type(exc).__name__,
                },
            )
            # Debug stack locations exclude exception messages, locals and bodies.
            logger.debug(
                "web_failure_stack %s",
                [(f.name, f.lineno) for f in traceback.extract_tb(exc.__traceback__)],
            )
        self.store.update(job)
