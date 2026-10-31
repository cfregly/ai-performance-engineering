# Draft-specific Auto-Plugs and publication results

Apply this policy to every new or revised social package. The author owns
publication. Draft preparation and read-only results tracking do not authorize
scheduling, publishing, sending a reply, or changing promotion on a live post.

## Prepare a useful plug for each eligible X draft

Every eligible original X draft, including a thread's first post, gets its own
Auto-Plug. Use the current draft's topic and the author's latest edits to offer one
useful next step. Retain the exact plug text and a verified, existing destination
that delivers that next step. Preserve the author's edits to an existing plug and
reconcile the latest readback before changing it. Use I and my for the author's own
work.

Do not reuse a universal repository pitch, invent new code or an unpublished
resource, or add a repository CTA to every post. A relevant existing article,
example, source or discussion can be a better destination. Keep the main post
useful on its own. When a link adds no value, a useful extra explanation or
concrete reader question can be the plug. Record `destination_reason` instead
of forcing a repository CTA. Record unsupported platforms or reply flows as
`not_applicable` with the exact reason. An inaccessible or paid-only setting is
a blocker, not proof that a usable plug was configured.

Prepare Auto-Retweet and Auto-Plug as dormant settings on the exact unscheduled
draft when supported and included in the current plan. Never change account
defaults, purchase an upgrade, or arm or revise promotion on an existing live
post during this workflow.

## Choose and retain the threshold

Use a cohort of comparable original X posts with known publication and
observation times. Start with the analytics workflow's latest 30 complete days
and retain all available relevant originals in that window. Paginate through
the provider's actual page limits. Deduplicate by native post ID. Include a
thread's first post once and exclude replies and later thread posts. Retain
weak results. Record the selection rule, included and excluded counts, actual
window, post ages, source, filters, capture time and raw metric definitions.
Missing data is unavailable, never zero. Keep LinkedIn evidence separate.

Choose the metric and threshold from the retained cohort. Record the actual
counts, sample size, observation ages and limitations in the private analytics
review. For sparse evidence, label the threshold provisional and explain the
operating choice. Do not treat the threshold as an established optimum or as
evidence that Auto-Plug improves reach. Keep account-specific metrics and
baseline sources out of this reusable policy.

Reassess after five comparable originals each have a snapshot taken at a post
age of at least seven days, or 30 days after the baseline observation, whichever
comes first. Record the decision, evidence and next review condition. Copy the
chosen metric, threshold and rationale into each draft's evidence and verify that draft's
actual setting. Do not change global defaults or retrospectively change live
posts. Keep threshold uncertainty and cohort bookkeeping in private evidence,
out of public copy.

## Verify the final draft

Before each write, confirm the current draft status and selected draft identity
against its visible post content. A URL navigation or a previously visible
editor alone does not prove that the next draft has loaded. Select the draft
card, confirm its selected state and actual post content, then open its
settings. If the author publishes during the work, freeze edits to that post and
continue only with publication/results readback.

After the last text, media, target or tag edit, reload the exact draft. Retain
the full readback and final `UNCHANGED` guard together with:

- Exact Auto-Plug text, verified destination and intended X account/target.
- Auto-Plug state, metric, numeric threshold and the trigger condition actually
  shown. Record Auto-Retweet state and trigger, capture time and evidence location.
- `status: draft` and null scheduling, publication, share and published URL/time
  fields, including platform-specific fields where exposed.

A saved copy or account default does not verify per-draft settings. Resolve
lost settings after edits and repeat the reload/readback. If safe readback is
unavailable, retain the blocker and leave the affected stage incomplete.

Retain `social/typefully-promotion-plan.json`, starting from
[the plan template](templates/typefully-promotion-plan.json), and declare it as
`promotion_plan` in the workflow receipt. The plan binds the exact saved reply,
threshold decision, native evidence and results registration. Validate it with
the workflow gate. Archived promotion receipts remain historical evidence. Do not use them to
restore an obsolete setting over a newer verified plan.

## Register publication and capture results

Register each prepared draft in `state/typefully-results/registry.json`, using
[the results template](templates/typefully-results.json) only to initialize a
missing central registry. Never replace an existing registry. A package's
`social/typefully-results.json` is a registration pointer: retain the registry
path, draft ID, social-set ID, platforms and `24h`/`7d` milestones there. Do not
copy the central ledger into every package. Use
`python3 tools/typefully_results.py --help` for the supported `track`, `ingest`,
`plan` and `snapshot` commands. The helper records evidence. It does not
publish or change account settings.

After the author publishes, discover publication through read-only Typefully/native
readback. Require the exact native URL and actual publication timestamp for
each platform. A planned or scheduled time, a draft URL, and a status label
alone are not publication evidence. X and LinkedIn can publish at different
times. Calculate each platform's age from its own actual timestamp.

Capture results at 24 hours and seven days, retaining the observation timestamp
and actual post age. For a late first observation, label its true age and the
missed window. Never manufacture a 24-hour snapshot. Retain the source/view,
definitions, raw evidence and available counts for impressions, likes or
reactions, replies/comments, reposts/reshares, bookmarks and clicks. Mark each
unavailable metric explicitly and explain access or source limits. Do not merge
X and LinkedIn totals or equate differently defined metrics.

Record the count of observed meaningful reader responses, their exact comment
links and a brief description of each question, correction or useful report.
Keep that reviewed count distinct from the platform's total reply count. For
Auto-Plug, record actual delivery state, exact reply URL and observed text and destination.
A configured setting does not prove delivery. Keep pending, not observed and
unavailable states distinct from a verified delivered reply.

Use the results to decide whether to retain or revise the hook, image,
destination or threshold for a future draft. Treat sparse comparisons as
descriptive and record alternative explanations. Do not claim causal lift.

## Keep monitoring read-only and quiet

The standalone helpers do not install a scheduler or monitor accounts. If a
separate read-only monitor is configured, check registered drafts for actual
publication and due observations. Keep notifications quiet while drafts are
unpublished or nothing actionable changed. Report a due capture, material
failure, or required user action. Retain capture provenance and prior results
so unchanged observations are not repeatedly reported.

Keep results tracking separate from any existing comment monitors. Those
monitors own contextual reply drafts. The results ledger can link existing
reply records, but it must not duplicate preparation, create reply drafts, or
send replies. No monitoring step authorizes installation, authentication,
purchases or account changes.
