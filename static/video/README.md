# Scene videos and provenance

The scene browser presents all **150 matched nuScenes validation scenes**, with **300 MP4s**: one original CMT prediction and one CMT + AsyncBEV prediction per scene. The source frame manifests identify the same ordered sample tokens and timestamps within each pair.

## Verified model mapping

| Website directory | Model | Original experiment directory |
| --- | --- | --- |
| `cmt_async_10` | Original CMT baseline | `cmt_sync_val_async_lidar_10` |
| `asyncbev_async_10` | CMT + AsyncBEV | `cmt_async_lidar_random_learned_vel_warping_with_loss_lr_2e-5_multi_modal_src_val_async_lidar_10` |

On 2026-10-18, the four previously published MP4s (scenes 0003 and 0016 for both models) were compared with the originals using SHA-256. All four matched byte-for-byte using the mapping above. The historical baseline path `scene-003` is normalized to `scene-0003`.

## Web copies

The 300 web MP4s total **758,645,764 bytes** (758.6 MB), keeping the static site within the [GitHub Pages 1 GB published-site limit](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits).

The original 300 MP4s total 6,090,106,955 bytes. They remain unchanged at the source. The website uses H.264/yuv420p copies at 1000 × 1000 pixels, x264 CRF 28 / slow (without B-frames), with MP4 faststart. All source frames and their variable-rate presentation timestamps are preserved. Some original container durations are inconsistent with those timestamps; the web copies use normalized container durations, with the original and web metadata recorded per file. Both web videos in each pair have the same duration. No model inference or visualization rerendering is involved. These lossy display copies are not evaluation inputs or new benchmark results.

`scenes.json` supplies the ordered IDs and paired URLs used by the page. `provenance.json` records each original file's experiment name, size, SHA-256, and metadata alongside its web copy. `SHA256SUMS` checks the web files. Absolute source paths and raw internal frame manifests are deliberately kept outside the published site.

When adding media, keep every scene available for both models, verify frame/timestamp correspondence, and regenerate all three manifests. Preserve original files and audit records. Check browser playback before publishing.
