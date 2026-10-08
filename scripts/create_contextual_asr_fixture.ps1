param(
    [string]$OutputDirectory = '.validation/phase8/benchmark-fixture',
    [string]$CasesPath = 'data/contextual_asr_benchmark.json'
)

# Original authored speech; local Windows SAPI synthesis, no external recording.
Add-Type -AssemblyName System.Speech
$contextFixtureRoot = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($contextFixtureRoot) | Out-Null
$contextCases = Get-Content -LiteralPath $CasesPath -Raw | ConvertFrom-Json
$contextSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $contextVoices = @($contextSynth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
    })
    if ($contextVoices.Count -lt 2) { throw 'Two English SAPI voices are required.' }
    $contextTurns = @()
    for ($contextIndex=0; $contextIndex -lt $contextCases.cases.Count; $contextIndex++) {
        $contextCase = $contextCases.cases[$contextIndex]
        $contextVoice = $contextVoices[$contextIndex % 2].VoiceInfo
        $contextSynth.SelectVoice($contextVoice.Name)
        $contextName = 'turn_{0:D4}.wav' -f ($contextIndex+1)
        $contextSynth.SetOutputToWaveFile((Join-Path $contextFixtureRoot $contextName))
        $contextSynth.Speak($contextCase.spoken_text)
        $contextSynth.SetOutputToNull()
        $contextTurns += [PSCustomObject]@{
            speaker_index=($contextIndex % 2); voice=$contextVoice.Name
            culture=$contextVoice.Culture.Name; text=$contextCase.spoken_text
            filename=$contextName; case_id=$contextCase.id; intended_term=$contextCase.term
        }
    }
    [PSCustomObject]@{
        origin='Original Phase VIII benchmark script synthesized locally with Windows SAPI'
        expected_speakers=2
        reference_status='Authored pronunciation/intended-term provenance; not verified acoustic ground truth'
        turns=$contextTurns
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $contextFixtureRoot 'provenance.json') -Encoding utf8
    Write-Output ('Created {0} original benchmark speech turns' -f $contextTurns.Count)
} finally { $contextSynth.Dispose() }
