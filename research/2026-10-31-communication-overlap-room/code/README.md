# Check Prime's reported step-time arithmetic

This offline example explains the relationship between throughput and step time using Mario Sieg and Prime Intellect's [published Mega MoE results](https://www.primeintellect.ai/blog/prime-mega-moe).

Requirements: Python 3.10 or newer. Checked with Python 3.14.6. No GPU, credentials, network access or extra packages are needed.

From this directory:

```sh
python3 report_prime_results.py
```

From the parent directory:

```sh
python3 code/report_prime_results.py
```

Within each model comparison, both runs process the same work per step: one packed 16,384-token row per GPU, no gradient accumulation and the same GPU count. Under those conditions, the step-time ratio `baseline / candidate` is also the throughput multiplier.

For Qwen3-30B-A3B, `2.78 / 2.08` gives about 1.337 times the throughput, a gain of about 33.7%. The time reduction is `(1 - 2.08 / 2.78) * 100`, about 25.2%. For Qwen3-235B-A22B, the same formulas give about 1.318 times the throughput, a gain of about 31.8%, and 24.1% less time. These calculations use the rounded times in [Prime's chart](https://www.primeintellect.ai/blog/prime-mega-moe/blog_step_time.png). Prime prints both throughput gains, 34% and 32%, and time reductions, 25% and 24%, in that chart. The helper explains those relationships without recovering unrounded timings.

The JSON also carries baseline and workload metadata. The script checks only the fields it uses: evidence class, unit, source URL, author, and each result's model, GPU count, times and reported gain. It does not validate or print the baseline, conditions, scope, source date or limits.

Use `--data PATH` for another file in this fixed article format. It must retain `evidence_class: source_reported`, `unit: seconds_per_step` and the linked Prime article's exact `source_url`. The author must be nonempty. Results need distinct model names, positive integer GPU counts and positive finite timings and gains. An empty results list is rejected.

Unreadable files, invalid JSON and rejected field values stop the report with exit code 1, no standard output and an error beginning `Cannot read source timings:`. These checks do not authenticate edited values against the source.

This example does not run a model, reproduce the benchmark, validate Prime's closed-source training kernel, or establish long-run training quality. The reported expert parallel exchange is inside each eight-GPU node. The [pinned integration documentation](https://github.com/PrimeIntellect-ai/prime-rl/blob/a566e5d8e2a903f44d901e114d5d510888b93303/docs/scaling.md#expert-parallelism) describes Prime's replacement kernel wheel. The arithmetic helper needs no such wheel.

Run the portable regression checks from the parent directory:

```sh
python3 code/test_report_prime_results.py
```

They check both printed ratios, default data lookup from another directory, help and input errors. Temporary edited inputs exercise wrong units, zero, oversized or nonfinite timings, and missing author credit. These checks validate the report, not Prime's benchmark.
