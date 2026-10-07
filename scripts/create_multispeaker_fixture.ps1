param([string]$OutputDirectory = '.validation/phase3/fixture')

# Original project text, synthesized locally. No identity inference or recording reuse.
# Invoke in Windows PowerShell with System.Speech available.
Add-Type -AssemblyName System.Speech
$fixtureRoot = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
$speakerSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speakerVoices = @($speakerSynth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
    })
    if ($speakerVoices.Count -lt 2) { throw 'Two distinct installed English SAPI voices are required.' }
    $speechTurns = @(
        @{speaker=0; text='The baseline uses a sixteen kilohertz waveform. I will run the benchmark tomorrow.'},
        @{speaker=1; text='Please measure latency and save the raw transcript. We need three reports by Friday.'},
        @{speaker=0; text='We should not change the original recording.'},
        @{speaker=1; text='Yes.'},
        @{speaker=0; text='Agreed. Let us check the speaker boundaries before we finish.'},
        @{speaker=1; text='I can review the timestamped words and the diarization turns.'}
    )
    $fixtureManifest = @()
    for ($turnIndex=0; $turnIndex -lt $speechTurns.Count; $turnIndex++) {
        $turn = $speechTurns[$turnIndex]
        $voice = $speakerVoices[$turn.speaker].VoiceInfo
        $speakerSynth.SelectVoice($voice.Name)
        $speakerSynth.Rate = 0
        $filename = 'turn_{0:D4}.wav' -f ($turnIndex+1)
        $speakerSynth.SetOutputToWaveFile((Join-Path $fixtureRoot $filename))
        $speakerSynth.Speak($turn.text)
        $speakerSynth.SetOutputToNull()
        $fixtureManifest += [PSCustomObject]@{
            speaker_index=$turn.speaker; voice=$voice.Name; culture=$voice.Culture.Name
            text=$turn.text; filename=$filename
        }
    }
    [PSCustomObject]@{
        origin='Original project script synthesized locally with Windows SAPI'
        expected_speakers=2
        reference_status='Synthetic voice/source provenance, not human-meeting benchmark annotations'
        turns=$fixtureManifest
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $fixtureRoot 'provenance.json') -Encoding utf8
    Write-Output ('Created {0} speech turns using {1} and {2}' -f $speechTurns.Count,$speakerVoices[0].VoiceInfo.Name,$speakerVoices[1].VoiceInfo.Name)
} finally { $speakerSynth.Dispose() }
