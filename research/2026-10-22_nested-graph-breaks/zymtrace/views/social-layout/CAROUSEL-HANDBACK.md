# LinkedIn profile carousel handback

Objective: preserve complete profile pages while keeping titles and footers clear of Typefully's top and bottom PDF viewer controls.

Status: COMPLETE for local assembly. No uploads, account actions, commits, installations, or GPU work were performed.

`linkedin-profiles.pdf` contains exactly four portrait pages. Each source SVG remains 1400 × 1400 pixels and is nested unchanged at y=175 on a 1400 × 1750 page, with 175px blank buffers above and below. PDF pages measure 1050 × 1312.5 points. No source content was rescaled or cropped.

Pages:

1. `linkedin-page-01.svg/png`: Nsys first post-warmup synchronized invocation.
2. `linkedin-page-02.svg/png`: NCU single-kernel metrics.
3. `linkedin-page-03.svg/png`: Zymtrace without nested graph-break support.
4. `linkedin-page-04.svg/png`: Zymtrace with nested graph-break support.

Every page has 400px and 520px previews. These portrait PNGs match the PDF geometry and are the page images to use for the package guard. Original Nsight square pages, X square cards, and portrait comparison remain unchanged. Native LinkedIn reader replies and copy banks can keep the original portrait comparison.

`carousel-manifest.json` records the source and output hashes, 1400 × 1750 page geometry, 175px content translation, scale 1, image bounds, translated text ink bounds, and PDF page dimensions. The helper `../render_profile_carousel.py` uses the existing bundled Python, pypdf, and `rsvg-convert -f pdf`. It reuses the existing source squares without rewriting them.

Validation passed: the output begins with `%PDF-`, contains exactly four portrait pages, and every nested SVG is structurally equal to its original after removing only its added x/y positioning attributes. Every text bounding box clears the first and last 120px of its page. The parent observed Typefully's controls covering about 116 source pixels. Both blank buffers exceed that area. All four final 400px portrait previews were inspected and show complete content without clipping. Detailed original Nsight report text requires opening the PDF at a larger size.

Blockers: none for local assembly. Parent owner must upload the revised PDF and verify the actual Typefully carousel with controls visible.

PDF SHA-256: `c094c4f72f5b05d907ecfd07c6e5bb30e8a0174b09d37eaf4ccdcf9b642a444b`.
