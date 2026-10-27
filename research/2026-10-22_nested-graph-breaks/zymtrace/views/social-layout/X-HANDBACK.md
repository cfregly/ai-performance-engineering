# X profile wrapper handback

Status: COMPLETE for local image wrappers. No account actions or source-card changes were performed.

Four 1428 × 1400 SVG and PNG wrappers are ready: `nsys-profile-x`, `ncu-profile-x`, `zymtrace-without-nested-graph-breaks-x`, and `zymtrace-with-nested-graph-breaks-x`. Each has 400px and 520px previews. The existing complete 1400px square source is translated to x=14, y=0 with no rescaling. Structural equality checks confirm that all source SVG content is preserved. Both Zymtrace sidecars are exact copies.

For the observed Typefully frame of 520 × 510, the predicted object-fit-cover loss is 0.03845%, below the 0.5% gate. `x-layout-manifest.json` records source/output hashes, dimensions, offset, scale, and crop calculation. `../render_x_profiles.py` reproduces the wrappers.

Original square cards, LinkedIn PDF, and portrait comparison remain unchanged. Parent owner can upload the four `-x.png` files and verify the final frame geometry.
