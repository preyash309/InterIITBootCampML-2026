param([string]$OutputDirectory = '.validation/phase6/fixture')

# Original project script; two local SAPI voices, no third-party recording.
Add-Type -AssemblyName System.Speech
$intelligenceFixtureRoot = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($intelligenceFixtureRoot) | Out-Null
$intelligenceSynth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $intelligenceVoices = @($intelligenceSynth.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
    })
    if ($intelligenceVoices.Count -lt 2) { throw 'Two English SAPI voices are required.' }
    $intelligenceTurns = @(
        @{speaker=0; text='For deployment, we could use cube net ease to manage the containers.'},
        @{speaker=1; text='Yes, agreed. We will use cube net ease for deployment.'},
        @{speaker=0; text='We could move the database to MongoDB, but this is only a proposal.'},
        @{speaker=1; text='No, we are not moving to MongoDB. Keep PostgreSQL as the database.'},
        @{speaker=0; text='Rahul, benchmark both models by Friday. Measure latency and memory usage.'},
        @{speaker=1; text='I will write the benchmark report. We will revisit timing later.'},
        @{speaker=0; text='We need to check whether Qdrant is faster. This work is confirmed, but no owner has been assigned.'},
        @{speaker=1; text='Someone should review the dashboard. That is a suggestion, not an assignment.'},
        @{speaker=0; text='The discount stays fifteen percent, not fifty percent. We are not approving a larger discount.'},
        @{speaker=1; text='Agreed. Keep the discount at fifteen percent. Do not deploy any database migration.'}
    )
    $intelligenceManifest = @()
    for ($intelligenceIndex=0; $intelligenceIndex -lt $intelligenceTurns.Count; $intelligenceIndex++) {
        $intelligenceTurn = $intelligenceTurns[$intelligenceIndex]
        $intelligenceVoice = $intelligenceVoices[$intelligenceTurn.speaker].VoiceInfo
        $intelligenceSynth.SelectVoice($intelligenceVoice.Name)
        $intelligenceFilename = 'turn_{0:D4}.wav' -f ($intelligenceIndex+1)
        $intelligenceSynth.SetOutputToWaveFile((Join-Path $intelligenceFixtureRoot $intelligenceFilename))
        $intelligenceSynth.Speak($intelligenceTurn.text)
        $intelligenceSynth.SetOutputToNull()
        $intelligenceManifest += [PSCustomObject]@{
            speaker_index=$intelligenceTurn.speaker; voice=$intelligenceVoice.Name
            culture=$intelligenceVoice.Culture.Name; text=$intelligenceTurn.text
            filename=$intelligenceFilename
        }
    }
    [PSCustomObject]@{
        origin='Original Phase VI meeting script synthesized locally with Windows SAPI'
        expected_speakers=2
        reference_status='Synthetic source/pronunciation provenance; not a human-meeting accuracy dataset'
        turns=$intelligenceManifest
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $intelligenceFixtureRoot 'provenance.json') -Encoding utf8
    Write-Output ('Created {0} original speech turns' -f $intelligenceTurns.Count)
} finally { $intelligenceSynth.Dispose() }
