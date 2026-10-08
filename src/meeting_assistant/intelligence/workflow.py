"""Developer I–VI orchestration; stage APIs and immutable artifacts remain separate."""

from dataclasses import replace
from pathlib import Path


def run_upstream(args, values):
    from meeting_assistant.asr import ASRConfig, transcribe_audio
    from meeting_assistant.asr.api_backend import WhisperAPIBackend
    from meeting_assistant.asr.serialization import save_transcript
    from meeting_assistant.audio import AudioIngestionConfig, ingest_audio
    from meeting_assistant.diarization import DiarizationConfig, PyannoteBackend
    from meeting_assistant.diarization.reconciliation import reconcile_transcript
    from meeting_assistant.diarization.serialization import save_speaker_transcript
    from meeting_assistant.grounding import (
        GroundingConfig,
        GroundingRetriever,
        ground_transcript,
        load_glossary,
    )
    from meeting_assistant.grounding.glossary import read_meeting_context
    from meeting_assistant.grounding.serialization import save_grounding
    from meeting_assistant.refinement import (
        GroqRefinerBackend,
        RefinementConfig,
        refine_transcript,
        save_refined_transcript,
    )

    asr_backend = WhisperAPIBackend(ASRConfig.from_env(environ=values))
    refiner_config = RefinementConfig.from_env(environ=values)
    refiner = GroqRefinerBackend(refiner_config)
    diar_config = DiarizationConfig.from_env(environ=values)
    diar_backend = PyannoteBackend(diar_config)
    glossary = load_glossary(
        project_glossary=args.project_glossary,
        meeting_entries=read_meeting_context(args.meeting_context) if args.meeting_context else (),
    )
    engine = GroundingRetriever(glossary, GroundingConfig.from_env(environ=values))
    context = getattr(args, "context_pack", None)
    if getattr(args, "context", None):
        from meeting_assistant.contextual_asr import read_context

        context = read_context(args.context)
    if context is not None:
        from meeting_assistant.contextual_asr.integration import context_retriever

        engine = context_retriever(context, engine)
    # Verify cached GPU/embedding prerequisites before paid speech upload.
    diar_backend.load_model()
    engine.prepare()
    options = {
        name: values[key]
        for name, key in (
            ("ffmpeg_path", "FFMPEG_PATH"),
            ("ffprobe_path", "FFPROBE_PATH"),
            ("work_dir", "AUDIO_WORK_DIR"),
        )
        if values.get(key)
    }
    audio = ingest_audio(args.input, config=replace(AudioIngestionConfig.from_env(), **options))
    output = args.output_dir or audio.canonical_audio_path.parent.parent / "transcripts"
    raw = transcribe_audio(audio.canonical_audio_path, backend=asr_backend)
    raw_files = save_transcript(raw, output)
    diarization = diar_backend.diarize(audio.canonical_audio_path)
    speaker = reconcile_transcript(raw, diarization, config=diar_config.reconciliation)
    speaker_files = save_speaker_transcript(speaker, diarization, output)
    from meeting_assistant.speaker_reliability.service import run_optional

    run_optional(
        audio.canonical_audio_path,
        diarization,
        speaker,
        output,
        environ=values,
        enabled=getattr(args, "speaker_reliability", False),
    )
    grounding = ground_transcript(speaker, retriever=engine)
    grounding_files = save_grounding(grounding, output)
    contextual = None
    if context is not None or getattr(args, "context_asr", False):
        from meeting_assistant.contextual_asr import (
            ContextualASRConfig,
            contextual_retranscribe,
            save_contextual_asr,
        )

        contextual_config = ContextualASRConfig.from_env(environ=values)
        if getattr(args, "context_asr", False):
            contextual_config = replace(contextual_config, enabled=True)
        contextual = contextual_retranscribe(
            audio.canonical_audio_path,
            speaker,
            grounding,
            context=context,
            config=contextual_config,
            asr_config=asr_backend.config,
        )
    kwargs = {"contextual_asr": contextual} if contextual is not None else {}
    try:
        refined = refine_transcript(
            speaker, grounding, backend=refiner, config=refiner_config, **kwargs
        )
    except Exception:
        if contextual is not None:
            save_contextual_asr(contextual, output, context=context)
        raise
    if contextual is not None:
        bundle = save_contextual_asr(contextual, output, context=context, refined=refined)
        print(
            f"Contextual ASR: {contextual.provider_calls} calls; {contextual.audio_seconds:.3f} s audio; {bundle}"
        )
    refined_files = save_refined_transcript(refined, output)
    print(f"Canonical WAV: {audio.canonical_audio_path}")
    for label, files in (
        ("Raw", raw_files),
        ("Speaker", speaker_files),
        ("Grounding", grounding_files),
        ("Refined", refined_files),
    ):
        print(f"{label} JSON: {files.json_path}")
    return refined, speaker, grounding, audio.canonical_audio_path, diarization, Path(output)
