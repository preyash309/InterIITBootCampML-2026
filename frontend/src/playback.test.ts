import { describe, expect, it } from "vitest";
import { PlaybackBoundary, timestamp } from "./playback";

describe("central evidence playback", () => {
  it("stops at the selected evidence end and does not stop twice", () => {
    const b = new PlaybackBoundary();
    b.set(22.48, 28.56);
    expect(b.tick(28.55)).toBe(false);
    expect(b.tick(28.56)).toBe(true);
    expect(b.tick(29)).toBe(false);
  });
  it("manual seeking cancels the evidence interval", () => {
    const b = new PlaybackBoundary();
    b.set(1, 2);
    b.clear();
    expect(b.tick(30)).toBe(false);
  });
  it("a new selection replaces the previous interval", () => {
    const b = new PlaybackBoundary();
    b.set(1, 2);
    b.set(3, 4);
    expect(b.tick(2)).toBe(false);
    expect(b.tick(4)).toBe(true);
  });
  it("rejects invalid intervals and rounds timestamps with carry", () => {
    expect(() => new PlaybackBoundary().set(3, 1)).toThrow();
    expect(timestamp(59.9999)).toBe("01:00.000");
  });
});
