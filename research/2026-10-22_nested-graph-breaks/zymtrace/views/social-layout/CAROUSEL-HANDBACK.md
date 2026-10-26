# LinkedIn profile carousel handback

Objective: retain Nsys, NCU, and both complete Zymtrace profiles in one LinkedIn PDF, with one profile view per page.

Status: COMPLETE for local assembly. No uploads, account actions, commits, installations, or GPU work were performed.

`linkedin-profiles.pdf` contains exactly four square pages, each 1050 × 1050 PDF points:

1. Retained Nsys first post-warmup synchronized invocation, fitted fully inside a square page.
2. Retained NCU single-kernel metrics, fitted fully inside a square page.
3. Complete existing square Zymtrace profile without nested graph-break support.
4. Complete existing square Zymtrace profile with nested graph-break support.

`nsys-profile-square` and `ncu-profile-square` each include a PNG, SVG, 400px preview, and 520px preview. Their headings are 56px, their arm labels are 44px, and their retained PNG bytes are embedded unchanged. The image aspect ratios are preserved without cropping. Original report values and screenshots were not redrawn. Their compact report text requires opening the PDF at a larger size for detailed reading.

`carousel-manifest.json` records source and output hashes, unchanged embedded-image method, image bounds, text ink bounds, and PDF page geometry. The renderer is `../render_profile_carousel.py` and uses the existing bundled Python with pypdf 6.10.0, plus `rsvg-convert -f pdf` for individual pages.

Validation passed: exactly four square PDF pages, embedded screenshot bytes match the retained source hashes, and every image/text bounding box fits inside its page. The two new PNG pages were inspected in full and at 400px and 520px. No clipping was observed. Existing Zymtrace cards remain unchanged.

Blockers: none for local assembly. Parent owner must upload the PDF and verify the actual Typefully LinkedIn carousel. Native LinkedIn reader replies and copy banks can keep the portrait comparison image.

PDF SHA-256: `397d3554ba3adb5fc0041968cb2f1a4b8566a7f7243b609d6d5cf74a967b7777`.
