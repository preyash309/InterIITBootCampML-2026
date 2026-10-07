param([string]$OutputDirectory = '.validation/phase4/fixture')

# Original, shareable project text. Spaced phonetic spellings direct SAPI
# pronunciation; ASR output is retained unchanged, never treated as ground truth.
Add-Type -AssemblyName System.Speech
$fixtureRoot = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($fixtureRoot) | Out-Null
$speakerSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $speakerVoices = @($speakerSynth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
    })
    if ($speakerVoices.Count -lt 2) { throw 'Two English SAPI voices are required.' }
    $speechTurns = @(
        @{speaker=0; text='We will use cue drant for vector search and pie torch to train the neural network.'},
        @{speaker=1; text='Deploy the containers with cube net ease. The NVIDIA GPU uses see you da kernels.'},
        @{speaker=0; text='For the chemical process, check the plug flow reactor and the distillation reflux ratio.'},
        @{speaker=1; text='Use the Newton Raphson method to solve the equation, then estimate the Reynolds number.'},
        @{speaker=0; text='Report annual recurring revenue and customer acquisition cost. The deadline is Friday, not Monday.'},
        @{speaker=1; text='The discount stays fifteen percent. Do not change the original transcript.'}
    )
    $fixtureManifest = @()
    for ($turnIndex=0; $turnIndex -lt $speechTurns.Count; $turnIndex++) {
        $turn = $speechTurns[$turnIndex]
        $voice = $speakerVoices[$turn.speaker].VoiceInfo
        $speakerSynth.SelectVoice($voice.Name)
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
        origin='Original Phase IV project script synthesized locally with Windows SAPI'
        expected_speakers=2
        reference_status='Synthetic pronunciation/source provenance, not a human-meeting accuracy dataset'
        turns=$fixtureManifest
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $fixtureRoot 'provenance.json') -Encoding utf8
    Write-Output ('Created {0} original speech turns' -f $speechTurns.Count)
} finally { $speakerSynth.Dispose() }
