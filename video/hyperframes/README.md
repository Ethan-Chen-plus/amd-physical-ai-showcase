# AMD Physical AI Demo Film

This directory is the reproducible HyperFrames source for the competition demo video.

## Render

```bash
cd /data/Data14TB/03competition/amd-physical-ai-showcase/video/hyperframes
npm install --ignore-scripts --no-fund --no-audit
bash render_delivery.sh
```

The render gate runs `lint`, `validate`, and `inspect` before producing a 1920x1080, 30 fps, 260 second master. Delivery outputs are copied to `assets/videos/`:

- `amd-physical-ai-demo-en.mp4`
- `amd-physical-ai-demo-zh.mp4`
- `amd-physical-ai-demo-bilingual.mp4`
- matching UTF-8 SRT files

The source of truth is `index.html`; `DESIGN.md`, `SCRIPT.md`, `STORYBOARD.md`, and `data/shot-manifest.json` document the style, narrative, timing, and evidence provenance.
