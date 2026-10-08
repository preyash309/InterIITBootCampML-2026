"""Local REST API with bounded uploads and safe access to existing phase artifacts."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException

from meeting_assistant.asr.config import read_environment
from meeting_assistant.asr.models import TranscriptResult
from meeting_assistant.asr.serialization import transcript_from_json
from meeting_assistant.diarization.models import SpeakerAwareTranscript
from meeting_assistant.diarization.serialization import speaker_transcript_from_json
from meeting_assistant.intelligence import (
    MeetingRecord,
    get_item_evidence,
    meeting_record_from_json,
)
from meeting_assistant.intelligence.exceptions import UnknownEvidenceError
from meeting_assistant.intelligence.models import EvidenceSpan
from meeting_assistant.refinement.models import RefinedTranscript
from meeting_assistant.refinement.serialization import refined_from_json

from .config import WebConfig
from .diagnostic_schemas import (
    ContextualASRResult,
    SemanticResult,
    Sidecar,
    SpeakerReliabilityResult,
)
from .diagnostics import DiagnosticReader
from .errors import WebError
from .jobs import JobWorker
from .locking import server_lock
from .orchestration import PipelineRunner
from .persistence import JobStore
from .registry import ARTIFACTS, safe_path
from .schemas import DownloadInfo, JobResponse

logger = logging.getLogger(__name__)


class SingleRecordingRoute(APIRoute):
    """Parse at most one file and no text fields before FastAPI resolves File()."""

    def get_route_handler(self):
        original = super().get_route_handler()

        async def limited(request: Request):
            async with request.form(max_files=1, max_fields=0):
                return await original(request)

        return limited


class BodyLimit:
    """Bound the incoming body before multipart spooling, including chunked requests."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        size = 0

        async def bounded_receive():
            nonlocal size
            message = await receive()
            size += len(message.get("body", b""))
            if size > self.limit:
                raise WebError(
                    "upload_too_large", "Recording exceeds the configured upload limit.", 413
                )
            return message

        try:
            headers = dict(scope["headers"])
            try:
                length = int(headers.get(b"content-length", b"0"))
            except ValueError as exc:
                raise WebError("invalid_request", "Invalid upload request.") from exc
            if length < 0:
                raise WebError("invalid_request", "Invalid upload request.")
            if length > self.limit:
                raise WebError(
                    "upload_too_large", "Recording exceeds the configured upload limit.", 413
                )
            await self.app(scope, bounded_receive, send)
        except WebError as exc:
            await JSONResponse({"error": exc.error.model_dump()}, status_code=exc.status)(
                scope, receive, send
            )


class UploadOriginGuard:
    """Reject cross-origin browser submissions before they can consume paid API quota."""

    def __init__(self, app, allowed):
        self.app, self.allowed = app, frozenset(allowed)

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in {"POST", "DELETE"}:
            origin = dict(scope["headers"]).get(b"origin")
            if origin is not None and origin.decode("latin-1") not in self.allowed:
                await JSONResponse(
                    {
                        "error": {
                            "code": "origin_denied",
                            "message": "Upload origin is not allowed.",
                        }
                    },
                    status_code=403,
                )(scope, receive, send)
                return
        await self.app(scope, receive, send)


def job_response(job):
    return JobResponse(
        **job.model_dump(),
        completed_stages=sum(s.status == "complete" for s in job.stages),
        status_url=f"/api/meetings/{job.id}/status",
        workspace_url=f"/meetings/{job.id}",
    )


def create_app(config=None, *, runner=None, values=None):
    values = read_environment(env_file=Path(".env")) if values is None else values
    config = config or WebConfig.from_env(values)
    store = JobStore(config.job_root)
    worker = JobWorker(store, runner or PipelineRunner(values))

    @asynccontextmanager
    async def lifespan(app):
        with server_lock(store.root):
            store.recover()
            worker.start()
            try:
                yield
            finally:
                if worker.thread.is_alive():
                    await asyncio.to_thread(worker.stop)

    app = FastAPI(title="Meeting assistant", version="0.1.0", lifespan=lifespan)
    app.state.store, app.state.worker, app.state.config = store, worker, config
    app.add_middleware(BodyLimit, limit=config.max_upload_bytes + 1024**2)
    app.add_middleware(
        UploadOriginGuard,
        allowed=config.allowed_origins
        + (f"http://127.0.0.1:{config.port}", f"http://localhost:{config.port}"),
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "Range"],
        expose_headers=["Content-Range", "Content-Disposition"],
        allow_credentials=False,
    )

    @app.exception_handler(WebError)
    async def web_error(request, exc):
        return JSONResponse({"error": exc.error.model_dump()}, status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def request_error(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "A meeting file or valid route parameter is required.",
                }
            },
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def malformed_request(request, exc):
        return JSONResponse(
            {
                "error": {
                    "code": "invalid_request",
                    "message": "Use one meeting file in the upload form."
                    if exc.status_code == 400
                    else "Requested resource is unavailable.",
                }
            },
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        logger.error("web_request_failed", extra={"exception_type": type(exc).__name__})
        return JSONResponse(
            {
                "error": {
                    "code": "server_error",
                    "message": "The server could not read or save this resource.",
                }
            },
            status_code=500,
        )

    uploads = APIRouter(route_class=SingleRecordingRoute)

    @uploads.post("/api/meetings", response_model=JobResponse, status_code=202)
    async def upload(file: UploadFile = File(...)):
        identifier = str(uuid4())
        directory = safe_path(store.root, identifier + "/upload", must_exist=False)
        temporary, source = directory / ".uploading", directory / "source.media"
        owned = False
        try:
            directory.parent.mkdir()  # Exclusive UUID claim; never reuse an existing job.
            owned = True
            directory.mkdir()
            size = 0
            with temporary.open("xb") as target:
                while block := await file.read(64 * 1024):
                    size += len(block)
                    if size > config.max_upload_bytes:
                        raise WebError(
                            "upload_too_large",
                            "Recording exceeds the configured upload limit.",
                            413,
                        )
                    target.write(block)
                target.flush()
                os.fsync(target.fileno())
            if not size:
                raise WebError("empty_upload", "Choose a nonempty meeting recording.", 400)
            os.rename(temporary, source)
            # Filename is bounded display metadata only, never a path or HTTP header.
            filename = "".join(c for c in (file.filename or "Recording") if c.isprintable())[:240]
            job = store.create(identifier, filename, size)
            response = job_response(job)
            worker.wake.set()
            return response
        except Exception:
            if owned:
                temporary.unlink(missing_ok=True)
                source.unlink(missing_ok=True)
                if directory.exists():
                    directory.rmdir()
                directory.parent.rmdir()
            raise
        finally:
            await file.close()

    @app.get("/api/meetings", response_model=list[JobResponse])
    def history():
        return [job_response(j) for j in store.list()]

    @app.delete("/api/meetings/{identifier}", status_code=204)
    def delete_failed(identifier: str):
        store.delete_failed(identifier)
        return Response(status_code=204)

    @app.get("/api/meetings/{identifier}", response_model=JobResponse)
    @app.get("/api/meetings/{identifier}/status", response_model=JobResponse)
    def status(identifier: str):
        return job_response(store.get(identifier))

    def completed(identifier):
        job = store.get(identifier)
        if job.status != "COMPLETED":
            raise WebError("meeting_not_ready", "This meeting has not completed processing.", 409)

    def load(identifier, kind, parser):
        return parser(store.artifact(identifier, kind).read_text(encoding="utf-8"))

    @app.get(
        "/api/meetings/{identifier}/contextual-asr", response_model=Sidecar[ContextualASRResult]
    )
    def contextual_asr(identifier: str):
        completed(identifier)
        return DiagnosticReader(store, identifier).get("contextual_asr")

    @app.get(
        "/api/meetings/{identifier}/speaker-reliability",
        response_model=Sidecar[SpeakerReliabilityResult],
    )
    def speaker_reliability(identifier: str):
        completed(identifier)
        return DiagnosticReader(store, identifier).get("speaker_reliability")

    @app.get(
        "/api/meetings/{identifier}/semantic-reasoning", response_model=Sidecar[SemanticResult]
    )
    def semantic_reasoning(identifier: str):
        completed(identifier)
        return DiagnosticReader(store, identifier).get("semantic_reasoning")

    @app.get("/api/meetings/{identifier}/record", response_model=MeetingRecord)
    def record(identifier: str):
        completed(identifier)
        return load(identifier, "meeting_json", meeting_record_from_json)

    @app.get("/api/meetings/{identifier}/transcript/raw", response_model=TranscriptResult)
    def raw(identifier: str):
        return load(identifier, "raw_json", transcript_from_json)

    @app.get("/api/meetings/{identifier}/transcript/speaker", response_model=SpeakerAwareTranscript)
    def speaker(identifier: str):
        return load(identifier, "speaker_json", speaker_transcript_from_json)

    @app.get("/api/meetings/{identifier}/transcript/refined", response_model=RefinedTranscript)
    def refined(identifier: str):
        return load(identifier, "refined_json", refined_from_json)

    @app.get("/api/meetings/{identifier}/evidence/{item_id}", response_model=list[EvidenceSpan])
    def evidence(identifier: str, item_id: str):
        completed(identifier)
        try:
            return get_item_evidence(
                load(identifier, "meeting_json", meeting_record_from_json),
                item_id,
                load(identifier, "refined_json", refined_from_json),
            )
        except UnknownEvidenceError as exc:
            raise WebError("unknown_evidence", "Evidence item not found.", 404) from exc

    @app.get("/api/meetings/{identifier}/audio")
    def audio(identifier: str):
        completed(identifier)
        return FileResponse(store.artifact(identifier, "canonical_audio"), media_type="audio/wav")

    @app.get("/api/meetings/{identifier}/downloads", response_model=list[DownloadInfo])
    def downloads(identifier: str):
        return [
            DownloadInfo(
                artifact=kind,
                **asdict(ARTIFACTS[kind]),
                download_url=f"/api/meetings/{identifier}/downloads/{kind}",
            )
            for kind in store.artifacts(identifier)
            if kind in ARTIFACTS
        ]

    @app.get("/api/meetings/{identifier}/downloads/{artifact}")
    def download(identifier: str, artifact: str):
        path = store.artifact(identifier, artifact)
        spec = ARTIFACTS[artifact]
        return FileResponse(path, media_type=spec.media_type, filename=spec.filename)

    app.include_router(uploads)

    if config.frontend_dist.is_dir():
        dist = config.frontend_dist.resolve()
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

        @app.get("/{route:path}", include_in_schema=False)
        def frontend(route: str):
            if route.startswith("api/") or route == "api":
                raise WebError("not_found", "API route not found.", 404)
            return FileResponse(dist / "index.html", media_type="text/html")

    return app
