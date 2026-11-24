# Issue tracker

Use [GitHub Issues](https://github.com/cfregly/ai-performance-engineering/issues) for issues, specs, and tickets. Target `cfregly/ai-performance-engineering` explicitly when using `gh`, including from a fork or a linked worktree.

**PRs as a request surface: no.**

## Conventions

- Read a ticket with `gh issue view <number> --repo cfregly/ai-performance-engineering --json number,title,body,labels,comments`.
- List issues with `gh issue list --repo cfregly/ai-performance-engineering --state open`. Filter with `--label` when needed.
- When a skill calls for publishing a spec or ticket, create a GitHub issue with `gh issue create --repo cfregly/ai-performance-engineering --title "..." --body-file <file>`.
- Apply the [triage labels](triage-labels.md) with `gh issue edit --add-label` or `--remove-label`. Use `gh issue comment --body-file` for comments and `gh issue close` to close an issue after recording the resolution.
- Link child tickets with `gh issue create --parent <parent>` or `gh issue edit <parent> --add-sub-issue <child>`.
- Use GitHub issue dependencies for blockers. Set them with `gh issue create --blocked-by <blocker>` or `gh issue edit <child> --add-blocked-by <blocker>`. Read `gh issue view <child> --json blockedBy` and count only nodes with `state` equal to `OPEN`. These sub-issue and dependency flags require `gh` 2.94 or later.

## Wayfinding

Use a GitHub issue labeled `wayfinder:map` for the map and native sub-issues for its tickets. Use `wayfinder:<type>` on each ticket, where `<type>` is `research`, `prototype`, `grilling`, or `task`. Create a missing Wayfinder label with `gh label create` when first using it. In map order, select the first open child that has no assignee and no open blocker. Claim it with `gh issue edit <number> --add-assignee @me`. Record the resolution on the child before closing it, then link that resolution from the map.
