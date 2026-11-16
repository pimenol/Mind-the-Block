---
description: Automatically commit and push changes at reasonable checkpoints during work.
---

# Git — Auto Commit & Push

## When to Commit

Commit after completing a logical unit of work:
- A new feature or module is working
- A bug fix or refactor is done
- Data processing or analysis step is complete
- Configuration or environment changes are applied
- Documentation updates

Do **not** batch unrelated changes into a single commit. Prefer small, focused commits.

## Commit Message Style

Use conventional-commit-style messages:

```
<type>: <short description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `data`, `docs`, `chore`, `test`, `style`

## Push

**Pushing directly to `main` is authorized.** This is a personal research repo — no PR
review is required. Push to `main` after committing.

## Workflow

```bash
git add <specific files>
git commit -m "<type>: <description>"
git push origin main
```

Always add specific files rather than `git add -A` to avoid accidentally committing secrets or large data files.
