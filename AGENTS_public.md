# Public agent guide for the AsyncBEV website

This guide applies to the static project website on `gh-pages`. Read [README.md](README.md) first. The CMT model implementation and its separate guide are on `main`.

## Source map

- `index.html` contains the research content, native More Research dropdown, and accessible scene-browser controls. Preserve the original UniBEV/VoteFlow links, authors, verified paper claims and template attribution. Keep `{AsyncBEV}` protected in both BibTeX title fields (`index.html` and `README.md`).
- `static/css/index.css` contains page styles; maintain desktop and mobile readability.
- `static/js/index.js` loads `static/video/scenes.json`, selects a pair and coordinates video playback. It also closes the native research dropdown on Escape, outside pointer activation, or focus moving away. Avoid preloading every scene.
- `tests/scene-browser.test.cjs` exercises manifest and playback state with mocked DOM/media using Node built-ins.
- `static/images/` contains the motivation figure, institution icon and original ICLR poster.
- `static/video/` contains 150 paired scenes, the scene manifest and media provenance/checksums. Read its [guide](static/video/README.md).

## Scientific and media conventions

Use the final ICLR 2026 paper as the content source. Distinguish paper-reported results from new measurements. The paper studies CMT and UniBEV; the released implementation and scene comparison use CMT. NDS differences in the highlights are percentage points, relative to the specified baseline and dynamic-object subset.

The CMT baseline belongs under `cmt_async_10`; the learned-flow AsyncBEV model belongs under `asyncbev_async_10`. SHA-256 comparisons established the original four website videos' source correspondence. Do not reverse these models based on a guessed filename mapping.

The website MP4s are compressed derivatives. Preserve original source videos separately, record both source and derivative hashes, preserve source frame counts/timestamps and paired duration, use H.264/yuv420p with faststart, and keep the published site below its hosting size limit. The canonical scene identifier uses four digits (e.g. `scene-0003`). Add paired manifest entries and update provenance/checksums whenever media changes. Never expose host paths, private logs, local agent instructions or discussion memory in public assets.

## Validation

Serve the site over HTTP, verify every listed video exists, and test initial load, previous/next/direct selection, play/pause, seeking, rate changes, replay, failure/retry and mobile overflow. Check the research dropdown with keyboard and touch, including Escape and outside dismissal; its links must remain usable without JavaScript. Verify the browser only requests selected videos and has no missing interpolation assets or JavaScript exceptions. Run `node --check static/js/index.js`, `node tests/scene-browser.test.cjs`, the SHA-256 check documented in README, and `git diff --check`.

Keep this guide and README current with changes to files, data conventions or validation. Preserve Git exclusions for private `AGENTS.md` and `.codex/`.
