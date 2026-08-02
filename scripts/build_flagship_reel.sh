#!/usr/bin/env bash
set -euo pipefail

# Build a web-safe success reel from public evidence clips.
# Each source is normalized before concatenation so the page has one stable format.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-$ROOT/assets/videos/flagship-reel.mp4}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/amd-public-reel.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

FONT="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
if [[ ! -f "$FONT" ]]; then
  FONT="/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"
fi

declare -a CLIPS=(
  "$ROOT/assets/videos/robocasa-gr00t-showcase-success.mp4|ROBOCASA365 · HOUSEHOLD SUCCESS"
  "$ROOT/assets/videos/dexjoco/recovery/bimanual-assembly.mp4|DEXJOCO · PI0.5 SUCCESS"
  "$ROOT/assets/videos/discoverse-showcase/cabinet-door-open-3view.mp4|DISCOVERSE · AMD 3-VIEW REPLAY"
  "$ROOT/assets/videos/discoverse-3dgs/franka-rocm-3dgs.mp4|3DGS · ROCm RENDERER"
)

concat_file="$TMP/concat.txt"
: > "$concat_file"
index=0
for item in "${CLIPS[@]}"; do
  src="${item%%|*}"
  label="${item#*|}"
  [[ -f "$src" ]] || { printf 'missing clip: %s\n' "$src" >&2; exit 2; }
  normalized="$TMP/clip-${index}.mp4"
  ffmpeg -hide_banner -loglevel error -y -i "$src" \
    -vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x0b0d10,fps=30,drawtext=fontfile=${FONT}:text='${label}':fontcolor=white:fontsize=28:box=1:boxcolor=0x111318cc:boxborderw=14:x=28:y=28" \
    -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$normalized"
  printf "file '%s'\n" "$normalized" >> "$concat_file"
  index=$((index + 1))
done

mkdir -p "$(dirname "$OUT")"
ffmpeg -hide_banner -loglevel error -y -f concat -safe 0 -i "$concat_file" \
  -an -c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUT"

ffprobe -v error -show_entries format=duration,size -show_entries stream=width,height,r_frame_rate,codec_name \
  -of default=noprint_wrappers=1 "$OUT"
printf 'Wrote %s\n' "$OUT"
