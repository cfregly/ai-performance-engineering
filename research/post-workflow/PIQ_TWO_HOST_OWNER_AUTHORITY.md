# PIQ two-host admitted owner-attested evidence

This public projection validates retained bytes only. It includes no real owner
callback or admitted result. The production admissions anchor is empty.
Validation does not independently prove execution, physical hosts, concurrency
or network traffic. Synthetic tests exercise this contract without admitting
production evidence.

The producer receipt at `evidence/two-host-functional-receipt.json` remains a
diagnostic retained-producer assertion. It can establish internal consistency
for a complete two-rank census, but producer labels do not establish physical
hosts, inter-host communication, owner custody or terminal cleanup.

An owner receipt can remove that producer receipt's one `READINESS_BLOCK` only
when both conditions hold:

1. The fixed package receipt and every referenced observation validate against
   this contract.
2. The receipt's exact owner/authority/producer/envelope/handback/run/attempt
   tuple appears in the fixed repository-owned admissions anchor after an
   independent source review.

**Hashes provide byte integrity, not origin authentication.** The admission is
an out-of-band source-review trust boundary, not a cryptographic signature.
The committed production list is intentionally empty until a real execution
owner callback is reviewed. If cryptographic origin authentication becomes a
requirement, define a separately reviewed signed version 2 rather than
weakening or overclaiming this version 1 contract.

## Fixed declarations and trust anchor

A schema-2 package declares both fixed paths:

```json
{
  "piq_two_host_functional_receipt": "evidence/two-host-functional-receipt.json",
  "piq_two_host_owner_authority_receipt": "evidence/two-host-owner-authority-receipt.json"
}
```

Use the receipt schema and template:

- [`schemas/piq-two-host-owner-authority-v1.schema.json`](schemas/piq-two-host-owner-authority-v1.schema.json)
- [`templates/two-host-owner-authority-receipt.json`](templates/two-host-owner-authority-receipt.json)

The validator loads admissions only from
[`trust/piq-two-host-owner-admissions.json`](trust/piq-two-host-owner-admissions.json),
validated by
[`schemas/piq-two-host-owner-admissions-v1.schema.json`](schemas/piq-two-host-owner-admissions-v1.schema.json).
There is no environment, command-line or package-provided override. Tests may
inject a trust document into the validator, but that is only test isolation and
does not admit production evidence.

Each source-reviewed admission has exact fields for `admission_id`, canonical
`admitted_at`, independent `reviewer_id`, retained source ID and hash,
`owner_id`, `authority_id`, producer-receipt hash, envelope hash,
owner-handback hash, `run_id`, and the ordered `attempt_ids`. Admission must
occur at or after the owner attestation. A different byte, identity, run or
attempt order does not inherit admission.

## Exact evidence boundary

The owner receipt binds exact producer-receipt bytes, the diagnostic envelope,
run ID, ordered selected attempts, owner handback, authority identity and
attempt records. It can remove only the producer readiness block. It cannot
suppress a producer error or any other package gate.

`authority.owner_id` and `authority.authority_id` are distinct, casefold-
distinct, specific portable ASCII identities, not generic labels such as
`owner`, `custodian`, `system` or `unknown`. All IDs are bounded and
control-free. `attested_at` is a canonical `Z` timestamp after every terminal
cleanup observation.

For each selected attempt, independently:

- The retained producer raw result supplies the native `startedAt` and
  `completedAt`. The owner launch exactly matches that native boundary and is
  inside the envelope objective window. Rank lifetimes are inside all three
  windows.
- Exactly ranks 0 and 1 bind their producer host and type-strict PID to the
  same producer census. Each `physical_host_id` exactly equals the canonical
  lowercase `execution.resources.cpuHosts[targetId].hostIdentitySha256` for
  that producer host. Producer labels cannot substitute for these hashes.
  Physical-host hashes and canonical lowercase UUID boot IDs are distinct
  under casefold comparison.
- Launch, controller and execution-session identities match across that
  attempt. Launch IDs and execution-session IDs may not be reused across
  selected attempts. A controller may be shared.
- One retained transport observation uses only `tcp`, `tcp4` or `tcp6`, joins
  ranks 0 and 1 with their exact PIDs, physical-host hashes and execution
  session, records positive bytes, and has equal sent/received payload hashes.
  Send and receive times are ordered inside both rank lifetimes. Transport IDs
  and the complete event identity/timing/payload fingerprint may not be reused
  across selected attempts. Local-only protocols are rejected.
- Exactly one cleanup observation per rank repeats that rank's physical host,
  boot, session, PID, launch/controller and transport identities. Each cleanup
  occurs after the launch ends, both ranks end and transport receipt, and says
  the PID is absent, transport is closed and session is terminal.

Each launch, rank, transport and cleanup observation is its own nonempty,
hashed regular JSON file below `evidence/`. It must be a nonsymlink file with
exactly one filesystem link and must equal the receipt's exact owner
projection. Duplicate JSON keys at any nesting level fail. References have
exactly `path` and `sha256`. Raw dot components, empty/double/trailing path
components, backslashes, wrong on-disk case, hash mismatches and reused
physical files fail. Malformed types fail closed without crashing. Evidence is
never unioned across attempts.

## Pure local handback and receipt builders

For a real owner callback, start with the full explicit facts template and
schema. The checked example is illustrative only:

- [`templates/two-host-owner-facts.json`](templates/two-host-owner-facts.json)
- [`schemas/piq-two-host-owner-facts-v1.schema.json`](schemas/piq-two-host-owner-facts-v1.schema.json)
- [`examples/piq-two-host-owner-facts/owner-facts.json`](examples/piq-two-host-owner-facts/owner-facts.json)

Build six observation documents per attempt plus the owner handback without
manual path/hash transcription:

```sh
python3 tools/build_piq_two_host_owner_handback.py /absolute/package \
  --producer-receipt evidence/two-host-functional-receipt.json \
  --facts evidence/owner-facts.json \
  --output-directory evidence/owner \
  --handback evidence/owner/execution-owner-handback.json
```

The builder takes only explicit local inputs, performs no runtime call or
authority inference, validates the staged result, installs all generated files
as one no-overwrite directory rename, and removes an unpublished stage on
failure. After the local output directory is installed, a durability or
final-verification failure preserves that directory for explicit inspection so
concurrent owner work is never recursively deleted. The builder does not create an admission.

Then assemble the fixed receipt:

```sh
python3 tools/assemble_piq_two_host_owner_receipt.py /absolute/package \
  --producer-receipt evidence/two-host-functional-receipt.json \
  --owner-handback evidence/owner/execution-owner-handback.json \
  --output evidence/two-host-owner-authority-receipt.json
```

This second builder also reads only explicit retained files, validates the
candidate, atomically refuses to overwrite its output and does not edit the
manifest. Source review and admission remain a separate repository-owner step.

Check the package with:

```sh
python3 tools/check_piq_two_host_functional.py /absolute/package
```

Producer-only validation intentionally exits 0 as a consistency-only API and
prints that readiness remains blocked. Once an owner receipt is declared, an
invalid or unadmitted receipt exits nonzero. Synthetic or locally assembled
evidence does not look ready.

## What an admitted pass means

An admitted pass checks the retained assertions and reviewed admission tuple
for the exact selected attempts. It does **not** independently establish
execution on physical hosts, scientific qualification, a speedup, repeatability,
a serving result, cryptographic origin authentication, current resource
authority or permission to publish.
The producer receipt remains `authority: "retained_producer_assertion"`, and
both receipts remain `qualification: "not_established"`. Scientific and
publication gates stay unchanged.
