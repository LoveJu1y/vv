# Security Policy

## Supported Scope

LaRA-VLA is a research codebase. The repository is maintained with a focus on:

- training and evaluation workflows
- config correctness
- removal of private defaults and credentials
- preserving third-party notices and provenance

It is not currently positioned as a hardened production deployment system.

## Reporting a Security Issue

If you believe you have found a security issue, please avoid opening a public issue with exploit details.

Instead, contact the maintainers privately using the repository contact channel or maintainer email listed in the project metadata when available.

When reporting, please include:

- a short description of the issue
- affected file(s) or script(s)
- steps to reproduce
- impact assessment
- whether any credentials, private paths, or remote-execution behavior are involved

## What Counts as a Security Issue

Examples include:

- accidentally committed secrets or credentials
- unsafe default behavior that exposes services unintentionally
- code paths that execute unexpected remote content by default
- path handling that leaks private local information in public defaults

## Response Expectations

Because this is a research repository, response times may vary. However, security-sensitive reports will be prioritized over feature requests.
