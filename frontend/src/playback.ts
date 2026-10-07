/** One bounded interval. Manual seeks clear it before the native player changes time. */
export class PlaybackBoundary {
  end: number | null = null;
  set(start: number, end: number) {
    if (
      !Number.isFinite(start) ||
      !Number.isFinite(end) ||
      start < 0 ||
      end <= start
    )
      throw new Error("Invalid playback interval");
    this.end = end;
  }
  clear() {
    this.end = null;
  }
  tick(time: number) {
    if (this.end !== null && time >= this.end) {
      this.clear();
      return true;
    }
    return false;
  }
}
export function timestamp(seconds: number) {
  const millis = Math.round(seconds * 1000);
  const hours = Math.floor(millis / 3600000);
  const mins = Math.floor(millis / 60000) % 60;
  const secs = Math.floor(millis / 1000) % 60;
  return `${hours ? String(hours).padStart(2, "0") + ":" : ""}${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(millis % 1000).padStart(3, "0")}`;
}
