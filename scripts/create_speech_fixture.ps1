param(
    [string]$OutputDirectory = '.validation/phase2/fixture'
)

# Run with Windows PowerShell 5.1: powershell.exe -File this-script.ps1.
# The words are original project text, spoken locally by an installed SAPI voice.
Add-Type -AssemblyName System.Speech
$fixtureDirectory = [System.IO.Path]::GetFullPath($OutputDirectory)
[System.IO.Directory]::CreateDirectory($fixtureDirectory) | Out-Null
$fixtureText = 'Today we are testing the meeting assistant. Please benchmark the speech recognition pipeline with sixteen thousand hertz audio. We need three results by Friday at ten thirty. Do not delete the original recording.'
$fixtureSynthesizer = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    $fixtureVoices = @($fixtureSynthesizer.GetInstalledVoices() | Where-Object {
        $_.Enabled -and $_.VoiceInfo.Culture.Name -like 'en-*'
    })
    if ($fixtureVoices.Count -eq 0) {
        throw 'An installed English SAPI voice is required to create this fixture.'
    }
    $fixtureSynthesizer.SelectVoice($fixtureVoices[0].VoiceInfo.Name)
    $fixtureSynthesizer.Rate = 0
    $fixtureSynthesizer.SetOutputToWaveFile((Join-Path $fixtureDirectory 'speech.wav'))
    $fixtureSynthesizer.Speak($fixtureText)
    $fixtureSynthesizer.SetOutputToNull()
    $fixtureText | Set-Content -LiteralPath (Join-Path $fixtureDirectory 'synthesis-source.txt') -Encoding utf8
    [PSCustomObject]@{
        origin = 'Original project text synthesized locally with Windows SAPI'
        voice = $fixtureSynthesizer.Voice.Name
        culture = $fixtureSynthesizer.Voice.Culture.Name
        text = $fixtureText
        reference_status = 'Synthesis source; listen and verify before treating as ground truth'
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $fixtureDirectory 'provenance.json') -Encoding utf8
    Write-Output "Fixture created in $fixtureDirectory"
} finally {
    $fixtureSynthesizer.Dispose()
}
