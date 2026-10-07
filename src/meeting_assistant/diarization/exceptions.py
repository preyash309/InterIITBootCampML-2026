"""Stable, sanitized failures for local diarization and evidence reconciliation."""


class DiarizationError(Exception):
    code = "diarization_error"


class DiarizationConfigurationError(DiarizationError):
    code = "diarization_configuration_error"


class DiarizationModelUnavailable(DiarizationError):
    code = "diarization_model_unavailable"


class DiarizationAccessError(DiarizationError):
    code = "diarization_access_error"


class DiarizationModelLoadError(DiarizationError):
    code = "diarization_model_load_error"


class DiarizationDeviceError(DiarizationError):
    code = "diarization_device_error"


class DiarizationInferenceError(DiarizationError):
    code = "diarization_inference_error"


class DiarizationOutOfMemory(DiarizationError):
    code = "diarization_out_of_memory"


class InvalidDiarizationAudio(DiarizationError):
    code = "invalid_diarization_audio"


class InvalidDiarizationResult(DiarizationError):
    code = "invalid_diarization_result"


class ReconciliationError(DiarizationError):
    code = "reconciliation_error"


class SpeakerTranscriptWriteError(DiarizationError):
    code = "speaker_transcript_write_error"
