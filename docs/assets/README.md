# Screenshot provenance

These four JPEG screenshots were selected from the local UI validation on
2026-10-07 and reviewed before publication. They contain no credentials, private
recordings or machine-specific paths. Total image size is about 245 KB.

| File | View | Origin |
|---|---|---|
| `upload.jpg` | Upload landing page | Empty local application |
| `overview.jpg` | Meeting summary and metrics | Original synthetic two-voice meeting |
| `transcript.jpg` | Anonymous speaker conversation | Same synthetic meeting |
| `evidence.jpg` | Actions, resolved evidence and bounded playback | Same synthetic meeting |

The original meeting script is in
[create_intelligence_fixture.ps1](../../scripts/create_intelligence_fixture.ps1).
It was synthesized locally with installed Windows SAPI English voices and
assembled using the project's fixture tooling. The named participant is fictional.
No third-party meeting audio was used. The generated WAVs, intermediate transcripts
and full screenshot collection remain ignored under `.validation/` and `jobs/`.

The displayed meeting was processed through the actual I–VI pipeline. These are
not mock inference screenshots. Recognition/correction errors such as `kube.net
ease` and `Drant` are intentionally visible; screenshot publication does not imply
perfect recognition or human-meeting benchmark accuracy.

See [browser validation](../phase7-validation.md) and
[UI validation](../ui-polish-validation.md) for the measured run and interactions.
