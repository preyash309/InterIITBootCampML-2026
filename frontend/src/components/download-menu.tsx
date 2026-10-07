import {
  ChevronDown,
  Download as DownloadIcon,
  FileText,
  Folder,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "./ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuSub,
  DropdownMenuSubContent,
  DropdownMenuSubTrigger,
  DropdownMenuTrigger,
} from "./ui/dropdown-menu";
import type { Download } from "../types";

const titles: { [key: string]: string } = {
  meeting_json: "Meeting record · JSON",
  meeting_md: "Minutes · Markdown",
  refined_json: "Refined transcript · JSON",
  refined_txt: "Refined transcript · Text",
  raw_json: "Raw transcript · JSON",
  raw_txt: "Raw transcript · Text",
  evidence_json: "Evidence manifest",
  speaker_json: "Speaker transcript · JSON",
  speaker_txt: "Speaker transcript · Text",
  diarization_json: "Diarization",
  grounding_json: "Grounding · JSON",
  grounding_txt: "Grounding · Text",
  edit_log_json: "Refinement edit log",
  canonical_audio: "Canonical audio · WAV",
};
/** Probe a single byte, then let the browser stream the original artifact download. */
export async function startDownload(file: Download) {
  try {
    const response = await fetch(file.download_url, {
      headers: { Range: "bytes=0-0" },
    });
    await response.body?.cancel();
    if (!response.ok) throw new Error("unavailable");
    const anchor = document.createElement("a");
    anchor.href = file.download_url;
    anchor.download = file.filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    toast.success("Download started", { description: file.filename });
  } catch {
    toast.error("Download unavailable", {
      description: "Check the server connection, then try again.",
    });
  }
}
export function DownloadMenu({ files }: { files: Download[] }) {
  const advanced = files.filter(
    (file) =>
      file.advanced ||
      ["canonical_audio", "speaker_json", "speaker_txt"].includes(
        file.artifact,
      ),
  );
  const order = [
    "meeting_json",
    "meeting_md",
    "refined_json",
    "refined_txt",
    "raw_json",
    "raw_txt",
    "evidence_json",
  ];
  const primary = files
    .filter((file) => !advanced.includes(file))
    .sort((a, b) => order.indexOf(a.artifact) - order.indexOf(b.artifact));
  const row = (file: Download) => (
    <DropdownMenuItem
      key={file.artifact}
      onSelect={() => void startDownload(file)}
    >
      <FileText />
      <span>{titles[file.artifact] ?? file.filename}</span>
    </DropdownMenuItem>
  );
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button variant="outline" className="download-trigger">
          <DownloadIcon />
          Downloads
          <ChevronDown />
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-72">
        <DropdownMenuLabel>Meeting artifacts</DropdownMenuLabel>
        {primary.map(row)}
        {!files.length && (
          <DropdownMenuItem disabled>No artifacts available</DropdownMenuItem>
        )}
        {!!advanced.length && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuSub>
              <DropdownMenuSubTrigger>
                <Folder className="mr-2 size-4" />
                Advanced artifacts
              </DropdownMenuSubTrigger>
              <DropdownMenuSubContent className="max-w-[calc(100vw-24px)]">
                {advanced.map(row)}
              </DropdownMenuSubContent>
            </DropdownMenuSub>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
