# AsyncBEV project website

Source for the [AsyncBEV project page](https://tudelft-iv.github.io/AsyncBEV/), published from the `gh-pages` branch of [tudelft-iv/AsyncBEV](https://github.com/tudelft-iv/AsyncBEV). The model implementation is maintained on `main`.

**AsyncBEV: Cross-modal Flow Alignment in Asynchronous 3D Object Detection**
Shiming Wang, Holger Caesar, Liangliang Nan, Julian F. P. Kooij · ICLR 2026

The page includes the abstract, method overview, paper and poster resources, and a synchronized browser for all 150 paired validation scenes (300 videos). Only the selected scene pair is loaded.

## Edit and preview

Read [AGENTS_public.md](AGENTS_public.md) before making changes. This is a static HTML/CSS/JavaScript site with no build step.

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open `http://127.0.0.1:8000/`. Python’s basic server is sufficient for layout preview; use a server with HTTP byte-range support for reliable seeking checks, as on GitHub Pages. Use an HTTP server because the scene selector fetches JSON; opening `index.html` directly as a file is insufficient.

- `index.html`: paper content and page structure.
- `static/css/index.css`: responsive layout and styling.
- `static/js/index.js`: scene selection and synchronized playback.
- `static/video/scenes.json`: ordered list of paired scenes and media URLs.
- `static/video/provenance.json`: source/web SHA-256, sizes, video metadata and encoding settings.
- `static/video/SHA256SUMS`: checksums for all 300 website MP4s.

See [video provenance](static/video/README.md) before replacing or adding media. Validate every manifest URL, both players, scene switching, seeking, rate changes, replay and mobile layout. Run `node --check static/js/index.js`, `node tests/scene-browser.test.cjs`, and `git diff --check`. The Node tests cover manifest validation and playback state with mocked media; browser checks verify actual decoding and interaction. Verify media integrity from `static/video/` with `sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS` on macOS).

## Citation

```bibtex
@inproceedings{wang2026asyncbev,
  author    = {Wang, Shiming and Caesar, Holger and Nan, Liangliang and Kooij, Julian F. P.},
  title     = {AsyncBEV: Cross-modal Flow Alignment in Asynchronous 3D Object Detection},
  booktitle = {International Conference on Learning Representations},
  year      = {2026},
  url       = {https://arxiv.org/abs/2601.12994}
}
```

## Website license

The website template is adapted from [Nerfies](https://github.com/nerfies/nerfies.github.io) under the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/).
