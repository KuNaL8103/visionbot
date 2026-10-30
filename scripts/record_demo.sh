#!/usr/bin/env bash
# Record a short, small GIF of the current X11 desktop for the README.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT_DIR}/docs/images/demo.gif"
DURATION="${1:-10}"
FPS="${FPS:-10}"
WIDTH="${WIDTH:-960}"
DISPLAY="${DISPLAY:-:0}"

command -v ffmpeg >/dev/null 2>&1 || {
  echo "ffmpeg is required (sudo apt install ffmpeg)" >&2
  exit 1
}

if [[ -n "${CAPTURE_SIZE:-}" ]]; then
  SIZE="$CAPTURE_SIZE"
elif command -v xdpyinfo >/dev/null 2>&1; then
  SIZE="$(xdpyinfo -display "$DISPLAY" | awk '/dimensions:/{print $2; exit}')"
elif command -v xrandr >/dev/null 2>&1; then
  SIZE="$(xrandr --current | awk '/\*/{print $1; exit}')"
else
  SIZE="1920x1080"
fi

mkdir -p "$(dirname "$OUT")"
echo "Recording ${SIZE} at ${FPS} fps for ${DURATION}s from ${DISPLAY}..."

ffmpeg -hide_banner -loglevel warning -y \
  -f x11grab -framerate "$FPS" -video_size "$SIZE" -i "$DISPLAY" \
  -t "$DURATION" \
  -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128:stats_mode=diff[p];[s1][p]paletteuse=dither=bayer:bayer_scale=5" \
  "$OUT"

SIZE_MB="$(du -m "$OUT" | awk '{print $1}')"
echo "Wrote ${OUT} (${SIZE_MB} MB)"
if (( SIZE_MB >= 10 )); then
  echo "GIF is >=10 MB; reduce duration, FPS, or WIDTH." >&2
  exit 1
fi
