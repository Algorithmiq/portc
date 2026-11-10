# Contributing to portc

Thanks for your interest in improving `portc`. Bug reports, benchmarks on
new hardware, and algorithmic improvements are all welcome.

## Contributor License Agreement

Before we can merge your contribution, you must sign the
[Contributor License Agreement](CLA.md). The CLA bot will post a one-time
comment on your first pull request with instructions; signing is a single
comment on that thread and applies to all your future contributions.

## Reporting bugs

Open an issue with:

- the versions of `portc`, `numpy` and (if relevant) `cupy`,
- a minimal reproducer, ideally with a fixed `seed=`,
- the observed and expected behaviour.

For numerical-accuracy reports, please include the bond dimensions, the number
of sites, the `dtype`, and the error metric you used.

## Development setup

The repository ships a [Dev Container](https://containers.dev/) configuration
that installs every dependency and the git hooks for you. If you would
rather set things up by hand, you need [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync --all-groups --all-extras
uv run prek install --prepare-hooks
```

If your clone still carries the old `pre-commit` shims, add `--force` to
replace them.

Agent instructions live in [`AGENTS.md`](AGENTS.md).
[Claude Code](https://code.claude.com/docs/en/memory#agents-md) supports this
file from v2.1.277; see its documentation for precedence and compatibility.

## Quality gates

All of these must pass before a pull request can be merged;
[`prek`](https://github.com/j178/prek) runs the lint, format and type checks automatically.

```bash
uv run ruff check src/ tests/       # lint
uv run ruff format --check src/ tests/
uv run ty check                     # type check src/ and tests/
uv run pytest -m "not slow"         # fast test suite
uv run pytest                       # full suite, including slow tests
```

New behaviour needs a test. Numerical changes need a test that pins the
accuracy, not just the shapes.

## Pull requests

- Keep pull requests focused on a single concern.
- Use [Conventional Commits](https://www.conventionalcommits.org/), optionally
  with a [gitmoji](https://gitmoji.dev/) after the colon, e.g.
  `fix: 🐛 trim terminal bond on the left-to-right sweep`.
- Update the documentation and the docstrings alongside the code.
- If the change affects performance, include before/after numbers and the
  hardware they were measured on.

## Code style

The project targets Python 3.11+ and is checked with `ruff` under a strict rule
set and type-checked with [`ty`](https://github.com/astral-sh/ty). Public functions carry type hints and Google-style docstrings without type
annotations in the argument list. Keep lines within the configured limit and
prefer clear code over clever code.
