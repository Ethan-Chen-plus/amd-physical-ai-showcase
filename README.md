# AMD Physical AI Showcase

Public evidence website for the Datawhale-EAI AMD Physical AI project.

The homepage defaults to English for competition review and has a Chinese/English toggle. The first viewport leads with the RoboCasa household-to-mobile-manipulation track, followed by successful RoboCasa, DexJoCo, DISCOVERSE, and 3DGS evidence. Failure and diagnostic clips are kept in the final section.

The site intentionally contains only public-safe assets:

- SHA-verified aggregate results;
- representative success and failure videos;
- public upstream, Notebook, and Hugging Face links;
- clearly separated formal, diagnostic, and in-progress claims.

The private competition adapters, credentials, machine addresses, raw datasets,
and unpublished training code are not included.

The migration evidence page is available at
`migrations.html`; it records the six-layer acceptance gates, DISCOVERSE and
RoboCasa365 evidence, policy boundaries, and the current DexJoCo status without
exposing private machine paths. The companion long-form Chinese engineering
blog is `migration-blog.html`; it explains the failure modes, repair order,
evaluation boundaries, and reproduction checklist in detail.

The RoboCasa365 gallery also includes a separate high-resolution, four-view
showcase rollout. It is presentation evidence only; the formal 16-task x
50-episode score remains unchanged.

`mobile-mainline.html` documents the RoboCasa365 mobile track: official mobile
task families, the verified PandaOmron 12-D action contract, recovery gates,
and the three release checks for data, policy, and publication. The AMD395
SmolVLA run is frozen at 5,000 steps and evaluated from the same
state16/action12 checkpoint on 3 tasks x 50 episodes: the real result is
`0 / 150`. The public page keeps this negative policy result separate from the
environment gate and links the checkpoint manifest, evaluation summary, and
representative videos.

The primary competition demo film is a 4:20, 1920x1080, 30fps HyperFrames
composition. It leads with DexJoCo and verified success footage, then moves
through RoboCasa365, DISCOVERSE, 3DGS, AMD execution evidence, and a failure
appendix. The homepage uses the bilingual MP4 as its primary player; English,
Chinese, bilingual, and standalone SRT deliverables are linked below the
flagship player. Release metadata and SHA-256 values are in
`data/amd-physical-ai-demo-release.json`.

The film is reproducible from the local HyperFrames source and FFmpeg helper:

```bash
HYPERFRAMES_BROWSER_PATH=/snap/bin/chromium \
  ./video/hyperframes/render_delivery.sh
```

The source storyboard, narration copy, shot manifest, and subtitles are kept
in `video/hyperframes/`.

The earlier public success reel remains reproducible with the local FFmpeg helpers:

```bash
scripts/build_flagship_reel.sh assets/videos/flagship-reel.mp4
scripts/burn_bilingual_subtitles.sh \
  assets/videos/flagship-reel.mp4 \
  assets/videos/flagship-reel.zh-en.srt \
  assets/videos/flagship-reel-bilingual.mp4
```

The source clips, bilingual subtitle file, generated MP4s, and `SHA256SUMS`
are kept together so the presentation layer remains traceable.

The DISCOVERSE section includes four isolated official MMK2 example replays on
AMD Radeon: box pick, drawer open, cabinet door open, and cup to plate. Each task has three native 1920x1080
camera videos (`cam_0`, `cam_1`, `cam_2`), plus a web-friendly three-view
composite and a SHA-256 result manifest. The composite is presentation-only;
it does not alter training data or formal evaluation denominators.

## Local preview

```bash
python3 -m http.server 4173 --directory .
```

Then open <http://127.0.0.1:4173/>.
