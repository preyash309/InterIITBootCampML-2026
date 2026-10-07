import { afterEach, describe, expect, it, vi } from "vitest";
import { api } from "../api";

class UploadRequest {
  static last: UploadRequest;
  upload = {
    onprogress: (_event: {
      lengthComputable: boolean;
      loaded: number;
      total: number;
    }) => {
      void _event;
    },
  };
  onload = () => {};
  onerror = () => {};
  responseText = "";
  status = 202;
  open = vi.fn();
  send = vi.fn();
  constructor() {
    UploadRequest.last = this;
  }
}
afterEach(() => vi.unstubAllGlobals());
describe("unchanged multipart upload contract", () => {
  it("sends one file, reports transferred bytes, and returns the accepted job", async () => {
    vi.stubGlobal("XMLHttpRequest", UploadRequest);
    const progress = vi.fn();
    const file = new File(["test"], "meeting.wav");
    const result = api.upload(file, progress);
    const xhr = UploadRequest.last;
    expect(xhr.open).toHaveBeenCalledWith("POST", "/api/meetings");
    expect((xhr.send.mock.calls[0][0] as FormData).get("file")).toBe(file);
    xhr.upload.onprogress({ lengthComputable: true, loaded: 5, total: 10 });
    expect(progress).toHaveBeenCalledWith(50);
    xhr.responseText = '{"id":"job_1"}';
    xhr.onload();
    expect(await result).toEqual({ id: "job_1" });
  });
  it("preserves the server's safe upload failure", async () => {
    vi.stubGlobal("XMLHttpRequest", UploadRequest);
    const result = api.upload(new File(["test"], "test.wav"));
    const xhr = UploadRequest.last;
    xhr.status = 413;
    xhr.responseText =
      '{"error":{"message":"Recording exceeds the upload limit."}}';
    xhr.onload();
    await expect(result).rejects.toThrow("Recording exceeds the upload limit.");
  });
  it("reports a network failure without pretending the upload completed", async () => {
    vi.stubGlobal("XMLHttpRequest", UploadRequest);
    const result = api.upload(new File(["test"], "test.wav"));
    UploadRequest.last.onerror();
    await expect(result).rejects.toThrow("Cannot reach the meeting server");
  });
});
