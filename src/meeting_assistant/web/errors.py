"""Application messages never interpolate upstream exceptions, paths or provider bodies."""

from starlette.exceptions import HTTPException

from .schemas import SafeError, StageName


class WebError(HTTPException):
    def __init__(self, code, message, status=400):
        self.error = SafeError(code=code, message=message)
        self.status = status
        super().__init__(status_code=status, detail=message)


def pipeline_error(exc, stage):
    code = getattr(exc, "code", "pipeline_error")
    # Only application-owned domain codes enter the response.
    if not type(exc).__module__.startswith("meeting_assistant."):
        code = "pipeline_error"
    if code in ("corrupt_media", "unsupported_media", "no_audio_stream", "empty_input_file"):
        message = "This recording is empty, damaged, unsupported, or has no readable audio track."
    elif "rate_limit" in code or getattr(exc, "status_code", None) == 429:
        message = "The transcription or language service quota is temporarily unavailable. Try a new upload later."
    elif "authentication" in code or "configuration" in code or "unavailable" in code:
        message = "This stage is not configured or available. Check the server configuration before uploading again."
    elif "model" in code or "device" in code or "memory" in code:
        message = "The local model could not run. Check cached models and GPU availability."
    else:
        label = {
            StageName.INGESTING: "Audio preparation",
            StageName.TRANSCRIBING: "Speech recognition",
            StageName.DIARIZING: "Speaker detection",
            StageName.GROUNDING: "Terminology grounding",
            StageName.REFINING: "Transcript refinement",
            StageName.EXTRACTING_INTELLIGENCE: "Meeting intelligence",
        }.get(stage, "Processing")
        message = f"{label} could not complete. Earlier artifacts have been retained. Check the server logs before a new upload."
    return SafeError(code=code, message=message)
