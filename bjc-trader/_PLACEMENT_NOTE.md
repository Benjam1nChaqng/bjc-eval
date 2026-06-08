# Placement note

This `bjc-trader/` directory is a **temporary holding location**. The project is a
standalone tool and is meant to live in its own dedicated repository (e.g.
`Benjam1nChaqng/bjc-trader`). It was committed here, onto the
`claude/advanced-stock-trading-bot-B9tNk` feature branch of `bjc-eval`, only because the
build session could not create a new GitHub repository.

It is isolated in this subdirectory and does not interact with `swe-judge`. To relocate:

```bash
# from a clean checkout
git subtree split --prefix=bjc-trader -b bjc-trader-export
# create the bjc-trader repo, then push that branch as its main
```

Do not merge this branch into `bjc-eval`'s main — `bjc-eval` is scope-frozen at v0.1
(ADR-005). Move this out to its own repo instead.
