# AMD Physical AI Showcase

Public evidence website for the Datawhale-EAI AMD Physical AI project.

The competition website is delivered entirely in English. The first viewport leads with the matched-protocol RoboCasa365 Pi0.5 and GR00T evaluation on AMD, followed by successful DexJoCo, DISCOVERSE, 3DGS, and PAC-MAN evidence. Failure and diagnostic clips are kept in the final section.

The public site includes:

- SHA-verified aggregate results;
- representative success and failure videos;
- public upstream, Notebook, and Hugging Face links;
- model and environment manifests for each published result.

The migration evidence page is available at
`migrations.html`; it records the six-layer acceptance gates, DISCOVERSE and
RoboCasa365 evidence, policy boundaries, and the current DexJoCo status without
exposing private machine paths. The companion long-form engineering
blog is `migration-blog.html`; it explains the failure modes, repair order,
evaluation boundaries, and reproduction checklist in detail.

The RoboCasa365 gallery also includes a separate high-resolution, four-view
showcase rollout. It is presentation evidence only; the formal 16-task x
50-episode score remains unchanged.

The RoboCasa365 benchmark is presented directly on the homepage. Official
Pi0.5 and GR00T checkpoints run the same 16-task, 50-episode-per-task protocol
on the AMD Ryzen AI MAX+ 395. The page links the aggregate JSON, per-task
comparison, synchronized four-view videos, and long-horizon success examples.

`perceptive-cbf-rl.html` documents the PAC-MAN AMD port. The upstream
predictive perpendicular CBF is validated with ROCm Torch and a portable CPU
MuJoCo proxy scene, producing a two-view MP4 and per-episode `eval_info.json`.
The page explicitly keeps the `12 / 12` proxy-safe result separate from the
upstream G1 AMP policy, MuJoCo-Warp training, and hardware benchmark. The
reproduction code and migration boundary are under
`code/perceptive_cbf_rl_amd/`.

The primary competition demo film is a 4:59, 1920x1080, 30fps HyperFrames
composition. It leads with DexJoCo and successful task footage, then moves
through RoboCasa365, DISCOVERSE, 3DGS, AMD execution evidence, and a failure
appendix. The homepage plays the English-narrated, English-captioned MP4. The
standalone English SRT deliverable is linked below the flagship player.
Release metadata and SHA-256 values are in
`data/amd-physical-ai-demo-release.json`.

The film is reproducible from the local HyperFrames source and FFmpeg helper. It
uses a Kokoro `af_heart` English voice, a deterministic original score, and a
generated editorial blueprint visual; no generated robot footage replaces the
real experiment clips:

```bash
HYPERFRAMES_BROWSER_PATH=/snap/chromium/current/usr/lib/chromium-browser/chrome \
HYPERFRAMES_WORKERS=2 \
  ./video/hyperframes/render_delivery.sh
```

The source storyboard, narration copy, shot manifest, and subtitles are kept
in `video/hyperframes/`.

The earlier public success reel remains reproducible with the local FFmpeg helper:

```bash
scripts/build_flagship_reel.sh assets/videos/flagship-reel.mp4
```

The source clips, generated MP4s, English captions, and `SHA256SUMS` are kept
together so the presentation layer remains traceable.

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
