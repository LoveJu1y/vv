# Contributing to LaRA-VLA

Thanks for your interest in improving LaRA-VLA.

## Before Opening a PR

- Check whether the change fits the current scope of the repository.
- Prefer small, focused pull requests over large mixed changes.
- If the change affects behavior, configs, or public interfaces, explain that clearly in the PR description.

## Development Expectations

- Keep user-facing text in English.
- Preserve existing third-party notices and file-level provenance headers.
- Avoid introducing private paths, credentials, or machine-specific defaults.
- Keep training, evaluation, and config paths aligned with the current `laravla/` namespace.

## Validation

Before submitting a PR, run the checks that are practical in your environment.

Typical checks include:

```bash
python -m py_compile $(find laravla deployment examples -type f -name '*.py')
bash -n scripts/run_bridge_multistage.sh
bash -n scripts/run_libero_multistage.sh
bash -n scripts/run_laravla_bridge.sh
bash -n scripts/run_laravla_libero.sh
```

If you change imports, configs, or launch scripts, include the exact validation you ran in the PR description.

## Pull Request Guidance

- Describe what changed and why.
- Call out any remaining risks or limitations.
- Mention whether the change affects training, evaluation, docs, or packaging.
- If a check could not be run, say so explicitly.

## Scope

This repository currently prioritizes:

- the `laravla` code namespace
- training entrypoints
- LIBERO and SimplerEnv evaluation flows
- concise documentation aligned with the current maintained workflow

Large refactors, new benchmarks, or major architectural additions are best discussed before implementation.
