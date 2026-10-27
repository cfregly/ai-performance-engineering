# LinkedIn profile carousel handback

Objective: keep complete profiles clear of Typefully top, bottom, left, and right viewer controls.

Status: COMPLETE for local assembly. No uploads, account actions, commits, installations, or GPU work were performed.

`linkedin-profiles.pdf` contains four square pages, each 1750 × 1750 pixels and 1312.5 × 1312.5 PDF points. Each unchanged 1400px square source SVG sits at x=175,y=175. Blank 175px margins on all four sides provide 55.4 CSS pixels at the observed 554px display width, clearing the observed controls up to 46 CSS pixels high and 44 pixels wide. Content remains at its original size without cropping or rescaling.

Pages and matching guard PNGs:

1. `linkedin-page-01.svg/png`: retained Nsys first post-warmup invocation.
2. `linkedin-page-02.svg/png`: retained NCU single-kernel metrics.
3. `linkedin-page-03.svg/png`: Zymtrace without nested graph-break support.
4. `linkedin-page-04.svg/png`: Zymtrace with nested graph-break support.

Each page has 400px and 520px previews. These square PNGs match the PDF geometry. Original Nsight square cards, X square cards, X-specific wrappers, and portrait comparison remain unchanged.

`carousel-manifest.json` records source/output hashes, 1750px square canvas geometry, content offset [175,175], scale 1, all text ink bounds, and PDF page sizes. `../render_profile_carousel.py` reproduces the pages and PDF with existing bundled pypdf and rsvg-convert.

Validation passed: PDF magic, exactly four square pages, exact source SVG structural equality after removing only added positioning attributes, matching 1750px square PNGs, and every text box clear of all four 140px edge regions. The 175px blank borders exceed the observed control footprint. Final previews were inspected. Detailed original Nsight report text requires viewing the PDF at a larger size.

Next step: parent owner uploads this final PDF and verifies the actual Typefully carousel with all controls visible.

PDF SHA-256: `13f3daf0e1ae2278725943cc6079cbbc91dd9ca22f020f217c4a40f3cc8f4a72`.
