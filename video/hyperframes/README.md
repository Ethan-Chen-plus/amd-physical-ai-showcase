# AMD Physical AI Demo Film

This directory is the reproducible HyperFrames source for the competition demo video.

## Render

```bash
cd /data/Data14TB/03competition/amd-physical-ai-showcase/video/hyperframes
npm install --ignore-scripts --no-fund --no-audit
bash render_delivery.sh
```

The render gate regenerates and validates phrase-aligned subtitles, then runs `lint`, `check`, and render for a 1920x1080, 30 fps, 299 second master. Subtitle timings are derived reproducibly from the actual chapter WAV pause boundaries and checked against the narration source. The audio step also fails if any narration gap exceeds 3 seconds. GSAP is pinned locally in `package.json`, and the delivery script prefers an installed Chromium binary so the render does not depend on a CDN or a browser download. Delivery outputs remain in `dist/`:

- `amd-physical-ai-demo-en.mp4`
- `amd-physical-ai-demo-zh.mp4`
- `amd-physical-ai-demo-bilingual.mp4`
- matching UTF-8 SRT files under `subtitles/`

The narration remains English in all film variants. The English film burns English captions. The Chinese and bilingual films burn one English line plus its Chinese translation for each spoken phrase. `subtitles/zh.srt` remains available as a Chinese-only sidecar. The delivery command does not modify site assets.

## Subtitle timing

```bash
npm run subtitles
npm run validate:subtitles
npm run qa:subtitles
```

`subtitles/cues.json` contains the English phrase and Chinese translation pairs. `scripts/generate_subtitles.py` detects natural pauses in each `audio/generated/<chapter>.wav`, selects ordered phrase boundaries, interpolates a timing boundary when a phrase has no measurable pause, verifies that the English cues reconstruct the narration text exactly, and writes all three SRT files. `scripts/validate_subtitles.py` checks language timing parity, phrase granularity, cue overlap, film bounds, reading speed, line count, and a strict 90-character English text budget. `scripts/render_subtitle_qa.sh` renders the first 10 seconds plus two review frames with the exact bilingual delivery style before a full render is started.

The source of truth is `index.html`; `DESIGN.md`, `SCRIPT.md`, `STORYBOARD.md`, and `data/shot-manifest.json` document the style, narrative, timing, and evidence provenance.
