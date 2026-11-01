# Typefully workflow for X and LinkedIn

Use this workflow for every new package and every revised social handoff. Its
goal is a useful, evidence-backed draft for the right readers. Analytics may
guide the hook, visual, or follow-up, but they do not prove causation, guarantee
reach, or relax the scientific gate. This guide covers draft preparation.
It does not replace complete package checks or technical reviews. See the
[snapshot guide](README.md) for the standalone offline tools and their limits.

Keep one execution owner for each draft. Use the installed Typefully interface
for supported draft actions. Resolve the configured social set, connected X and
LinkedIn identities, and existing tags before editing. An authentication failure
stops account work: retain the blocker and do not reconnect, install, upgrade,
or configure an integration. Never retain credentials in evidence.

## 1. Review relevant analytics

Copy [the analytics review](templates/typefully-analytics-review.md) to
`social/typefully-analytics-review.md`. Use the latest 30 complete days and all
available relevant posts in that window. Record the actual window,
timezone, source, selection rule, post count, post ages, extraction time, and
account. Include weak results rather than selecting only winners.

Keep X and LinkedIn sources and metric definitions separate. Show raw counts
and denominators beside rates. Do not calculate a rate when its denominator is
missing or zero. Treat comparisons from a small or selected sample as
descriptive. If a view is unavailable, record the specific blocker and proceed
with explicitly labeled editorial judgment. This review never changes a
schedule or publishes an experiment.

## 2. Review relevant people and source conversations

Copy [the people review](templates/typefully-people-review.md) to
`social/typefully-people-review.md`. Begin with authors, maintainers, and
engineers named by the package sources, then inspect only directly relevant
prior engagement and original posts. Research every person whose material
contribution the draft uses. Keep only relevant people and conversations in
the shortlist, with the contribution and verification evidence for each. Follow
[the share-optimization check](SHARE_OPTIMIZATION.md). An empty shortlist is
valid only after the identity search and its result are recorded.

Verify each profile and native target. Never infer an X handle or LinkedIn URN
from a name. Plain `@name` text is not proof of a native mention. Include a
mention only when it provides useful attribution or directly answers that
person. Preserve the exact native parent for a reply or quote. Discovery and
local drafting authorize no message, mention on a live post, reply, quote,
repost, or reshare.

Every package also keeps `source-comment.md`, starting from
[the source-comment template](templates/source-comment.md). It is a useful
comment for the source-native context, not a disguised promotion. Keep it
explicitly draft and unsent until separate authorization.

## 3. Preserve lifecycle and mention state

Use the social set's existing lifecycle tag slugs. Do not create a new taxonomy.

| Existing slug | Use |
|---|---|
| `idea` | An undeveloped topic or opportunity. |
| `drafting` | Copy, evidence, or images are still being prepared. |
| `needs-review` | The prepared draft is ready for the author's content review. |
| `approved` | The author explicitly approved this exact content; retain that approval. |

Keep exactly one lifecycle tag and preserve unrelated existing tags. A material
edit returns the draft to `drafting` or `needs-review`. Record native mention
state as `none`, `verified`, `unresolved`, or `partial`, with the retained people
review. No tag or mention state authorizes publication, scheduling, promotion on
a live post, or contact with another person.

## 4. Prepare and read back the exact unscheduled draft

Use the [draft guard](tools/typefully_draft_guard.py). Reuse an
unambiguous matching draft. Before changing it, compare the current text, media,
targets, platform settings, and tags with the last exact receipt. Preserve
manual edits and comment anchors. Keep the main X and LinkedIn package in one
draft; use separate drafts for distinct reply or quote targets.

The installed CLI text-update path was observed to discard X platform settings,
including `reply_to_url`, on October 31. For an existing reply, retain the exact
parent before editing. After saving, compare settings as well as text. If the
parent is lost, stop the update sequence and reconcile the write. Restore the
exact parent through the native “Reply or continue thread” control, then
require a new `UNCHANGED` guard. Do not create a replacement draft or label a
standalone post as a prepared reply.

After the final edit, reload and retain the full detail readback. X and LinkedIn
must both remain enabled. Save their exact post boundaries, text, per-post media
IDs, target and platform settings, lifecycle and unrelated tags, and remote
status. Run `tools/typefully_draft_guard.py` again and retain its `UNCHANGED`
result. A write response, upload response, or stale guard result is insufficient.

The final state is a plain unscheduled draft: `status` is `draft`; scheduling,
publication, share, and published-URL/time fields are null; and publication is
not authorized. Do not call a publishing or scheduling endpoint to test support.

## 5. Verify dormant promotion settings

Follow [the draft-specific Auto-Plug and results policy](AUTOPLUG_AND_RESULTS.md).
Every eligible original X draft needs a useful plug tied to its current topic,
an exact verified destination, and its own evidence-based threshold. Retain
the bounded cohort and threshold decision in the private analytics review.

Inspect Auto-Retweet and Auto-Plug after the final text, media, target, and tag
edit, then reload and inspect again. Record the actual `on`, `off`, or
`not_applicable` state, trigger or reason, observation time, and retained
evidence. An `on` setting must remain dormant while the draft is unscheduled.
Do not infer a per-draft state from an account default.

In the native UI observed on October 31, open **Review and publish draft** to
inspect **Auto-RT** and **Auto-Plug**, then close the review dialog. Never press
**Publish now** during draft preparation. **Spread the word** inserts a
Typefully signature post. It is not a promotion-settings control. If an
unintended change occurs, retain the exact before/after diff, restore only the
known change against the saved draft baseline, and verify the full guard.

After each reload, select the exact draft and verify its ID in Draft Properties.
Before saving a phone preview, verify the selected platform and require every
uploaded-image URL to contain that draft's expected media ID with a loaded
source image. A title appearing elsewhere in the draft list does not identify
the active editor. Retain a wrong-draft capture as rejected evidence and replace
it. Never bind it to the package's passing preview receipt.

Never change a global default, buy an upgrade, enable paid overage, or arm or
change promotion on a live post during draft-only work. If a setting cannot be
checked safely, keep the package incomplete rather than calling it enabled.

## 6. Retain the workflow receipt

Declare `typefully_workflow_receipt: "social/typefully-workflow.json"` in the
schema-2 manifest and start from
[the workflow template](templates/typefully-workflow.json). Manifest schema 1
is retained compatibility only; every new package uses schema 2. Migrating an
older package requires all schema-2 artifacts together.
Hash-bind the final raw readback, final `UNCHANGED` guard result, analytics
review, people review, and promotion evidence. For an update, also retain the
prior raw readback; the gate checks draft identity and preservation of unrelated
tags. An `approved` tag requires hashed evidence of the author's approval of that
exact copy.

Retain the exact plug text, destination, per-draft metric/threshold and final
reload evidence. Prepare the separate results record under
[AUTOPLUG_AND_RESULTS.md](AUTOPLUG_AND_RESULTS.md) so read-only tracking can
start from each platform's actual native URL and publication timestamp after
the author publishes. This record does not authorize publication or live promotion.

Run the package gate after the receipt is complete. Report the draft URL,
lifecycle tag, actual promotion states and triggers, verified or unresolved
mentions, proposed source-conversation targets, and access blockers. This
workflow creates no scheduler and sends or publishes nothing.
