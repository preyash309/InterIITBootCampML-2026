"""Stable application-facing errors; diagnostics are kept separate from messages."""


class AudioIngestionError(Exception):
    code = "audio_ingestion_error"

    def __init__(self, message: str, *, stderr: str = "", returncode: int | None = None) -> None:
        super().__init__(message)
        self.stderr = stderr
        self.returncode = returncode


class ConfigurationError(AudioIngestionError):
    code = "invalid_configuration"


class InputFileNotFound(AudioIngestionError):
    code = "input_file_not_found"


class InvalidInputFile(AudioIngestionError):
    code = "invalid_input_file"


class EmptyInputFile(AudioIngestionError):
    code = "empty_input_file"


class InputTooLarge(AudioIngestionError):
    code = "input_too_large"


class FileAccessError(AudioIngestionError):
    code = "file_access_error"


class UnsupportedMedia(AudioIngestionError):
    code = "unsupported_media"


class CorruptMedia(AudioIngestionError):
    code = "corrupt_media"


class NoAudioStream(AudioIngestionError):
    code = "no_audio_stream"


class FFmpegUnavailable(AudioIngestionError):
    code = "ffmpeg_unavailable"


class FFprobeUnavailable(AudioIngestionError):
    code = "ffprobe_unavailable"


class ExternalCommandTimeout(AudioIngestionError):
    code = "external_command_timeout"


class ExternalCommandFailed(AudioIngestionError):
    """Internal wrapper error translated at the inspection/conversion boundary."""

    code = "external_command_failed"


class MetadataExtractionFailed(AudioIngestionError):
    code = "metadata_extraction_failed"


class AudioConversionFailed(AudioIngestionError):
    code = "audio_conversion_failed"


class InvalidCanonicalAudio(AudioIngestionError):
    code = "invalid_canonical_audio"


class InvalidJobId(AudioIngestionError):
    code = "invalid_job_id"


class JobBusy(AudioIngestionError):
    code = "job_busy"
