param([string]$OutputDirectory = '.validation/phase9/fixtures')
$ErrorActionPreference = 'Stop'
# SAPI event times are authored source-word probes, not forced-aligned ASR words.
Add-Type -AssemblyName System.Speech
Add-Type -ReferencedAssemblies @('System.Speech', 'System.Collections', 'System.Runtime') -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.Speech.Synthesis;
public class Phase9WordEvent {
    public double time_seconds;
    public string text;
    public int character_position;
}
public class Phase9Synthesis {
    public static List<Phase9WordEvent> Save(string voice, string text, string path) {
        var words = new List<Phase9WordEvent>();
        using (var synth = new SpeechSynthesizer()) {
            synth.SelectVoice(voice);
            synth.SpeakProgress += (sender, e) => {
                words.Add(new Phase9WordEvent {time_seconds=e.AudioPosition.TotalSeconds,
                    text=e.Text, character_position=e.CharacterPosition});
            };
            synth.SetOutputToWaveFile(path);
            synth.Speak(text);
        }
        return words;
    }
}
'@
$phase9Root = [IO.Path]::GetFullPath($OutputDirectory)
[IO.Directory]::CreateDirectory($phase9Root) | Out-Null
$phase9Cases = Get-Content -LiteralPath data/speaker_reliability_fixtures.json -Raw | ConvertFrom-Json
foreach ($phase9Case in $phase9Cases.fixtures) {
    $phase9CaseRoot = Join-Path $phase9Root $phase9Case.id
    [IO.Directory]::CreateDirectory($phase9CaseRoot) | Out-Null
    $phase9Turns = @()
    for ($phase9Index=0; $phase9Index -lt $phase9Case.turns.Count; $phase9Index++) {
        $phase9Turn = $phase9Case.turns[$phase9Index]
        $phase9Voice = $phase9Cases.voices[[int]$phase9Turn[0]]
        $phase9Name = 'turn_{0:D4}.wav' -f ($phase9Index+1)
        $phase9Words = [Phase9Synthesis]::Save($phase9Voice, $phase9Turn[1], (Join-Path $phase9CaseRoot $phase9Name))
        $phase9Turns += [PSCustomObject]@{speaker_index=$phase9Turn[0]; voice=$phase9Voice;
            text=$phase9Turn[1]; filename=$phase9Name; word_events=@($phase9Words)}
    }
    [PSCustomObject]@{origin=$phase9Cases.origin; id=$phase9Case.id;
        gap_seconds=$phase9Case.gap_seconds; overlap_seconds=$phase9Case.overlap_seconds;
        turns=$phase9Turns} | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $phase9CaseRoot 'provenance.json') -Encoding utf8
}
Write-Output 'Synthesized six original controlled fixtures with source-word event times.'
