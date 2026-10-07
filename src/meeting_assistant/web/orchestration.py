"""Calls existing public stage APIs; cached model instances belong to one worker."""

from dataclasses import replace

from .schemas import StageName as S


class PipelineRunner:
    def __init__(self, values):
        self.values = dict(values)
        self.backends = None

    def _models(self):
        if self.backends is None:
            from meeting_assistant.asr import ASRConfig
            from meeting_assistant.asr.api_backend import WhisperAPIBackend
            from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend
            from meeting_assistant.grounding import (
                GroundingConfig,
                GroundingRetriever,
                load_glossary,
            )
            from meeting_assistant.intelligence import (
                GroqMeetingIntelligenceBackend,
                IntelligenceConfig,
            )
            from meeting_assistant.refinement import GroqRefinerBackend, RefinementConfig

            dc = DiarizationConfig.from_env(environ=self.values)
            rc = RefinementConfig.from_env(environ=self.values)
            ic = IntelligenceConfig.from_env(environ=self.values)
            diarizer = PyannoteBackend(dc)
            engine = GroundingRetriever(
                load_glossary(), GroundingConfig.from_env(environ=self.values)
            )
            # Fail local prerequisites before spending API quota, as in existing CLI.
            diarizer.load_model()
            engine.prepare()
            self.backends = (
                WhisperAPIBackend(ASRConfig.from_env(environ=self.values)),
                diarizer,
                dc,
                engine,
                GroqRefinerBackend(rc),
                rc,
                GroqMeetingIntelligenceBackend(ic),
                ic,
            )
        return self.backends

    def run(self, identifier, source, root, execute):
        from meeting_assistant.asr import transcribe_audio
        from meeting_assistant.asr.serialization import save_transcript
        from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
        from meeting_assistant.diarization import reconcile_transcript, save_speaker_transcript
        from meeting_assistant.grounding import ground_transcript
        from meeting_assistant.grounding.serialization import save_grounding
        from meeting_assistant.intelligence import extract_meeting_record, save_meeting_record
        from meeting_assistant.refinement import refine_transcript, save_refined_transcript

        output = root / identifier / "transcripts"

        def ingestion():
            config = AudioIngestionConfig.from_env()
            conversions = {
                "AUDIO_FFMPEG_TIMEOUT": ("ffmpeg_timeout_seconds", float),
                "AUDIO_FFPROBE_TIMEOUT": ("ffprobe_timeout_seconds", float),
                "AUDIO_MAX_INPUT_SIZE": ("max_input_size_bytes", int),
                "AUDIO_MIN_DURATION": ("min_audio_duration_seconds", float),
                "AUDIO_DURATION_TOLERANCE_SECONDS": ("duration_tolerance_seconds", float),
                "AUDIO_DURATION_TOLERANCE_RATIO": ("duration_tolerance_ratio", float),
                "FFMPEG_PATH": ("ffmpeg_path", str),
                "FFPROBE_PATH": ("ffprobe_path", str),
            }
            options = {
                name: parser(self.values[key])
                for key, (name, parser) in conversions.items()
                if self.values.get(key)
            }
            audio = ingest_audio(
                source, config=replace(config, work_dir=root, **options), job_id=identifier
            )
            return audio, {"canonical_audio": audio.canonical_audio_path}

        audio = execute(S.INGESTING, ingestion)

        def asr():
            backend, *_ = self._models()
            raw = transcribe_audio(audio.canonical_audio_path, backend=backend)
            files = save_transcript(raw, output)
            return raw, {"raw_json": files.json_path, "raw_txt": files.text_path}

        raw = execute(S.TRANSCRIBING, asr)
        _, diarizer, dc, engine, refiner, rc, intelligence, ic = self._models()

        def diarization():
            diar = diarizer.diarize(audio.canonical_audio_path)
            speaker = reconcile_transcript(raw, diar, config=dc.reconciliation)
            files = save_speaker_transcript(speaker, diar, output)
            return (diar, speaker), {
                "diarization_json": files.diarization_path,
                "speaker_json": files.json_path,
                "speaker_txt": files.text_path,
            }

        diar, speaker = execute(S.DIARIZING, diarization)

        def grounding():
            result = ground_transcript(speaker, retriever=engine)
            files = save_grounding(result, output)
            return result, {"grounding_json": files.json_path, "grounding_txt": files.text_path}

        grounded = execute(S.GROUNDING, grounding)

        def refinement():
            result = refine_transcript(speaker, grounded, backend=refiner, config=rc)
            files = save_refined_transcript(result, output)
            return result, {
                "refined_json": files.json_path,
                "refined_txt": files.text_path,
                "edit_log_json": files.edit_log_path,
            }

        refined = execute(S.REFINING, refinement)

        def extraction():
            result = extract_meeting_record(
                refined,
                speaker,
                grounded,
                backend=intelligence,
                config=ic,
                canonical_audio_path=audio.canonical_audio_path,
                diarization=diar,
            )
            files = save_meeting_record(result, refined, output)
            return result, {
                "meeting_json": files.json_path,
                "meeting_md": files.markdown_path,
                "evidence_json": files.evidence_manifest_path,
            }

        execute(S.EXTRACTING_INTELLIGENCE, extraction)
