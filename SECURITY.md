# Security Policy

## Supported versions

Only the latest released version of `portc` receives security updates.

## Reporting a vulnerability

Please do **not** open a public issue for security problems.

Report vulnerabilities privately through
[GitHub's private vulnerability reporting](https://github.com/Algorithmiq/portc/security/advisories/new),
or by email to <aurora@algorithmiq.fi>.

Include a description of the issue, the affected version, and a reproducer if
you have one. We aim to acknowledge reports within five working days and to
publish a fix and an advisory once the issue is confirmed and resolved.

## Scope

`portc` is a numerical library: it performs tensor-network arithmetic on
arrays supplied by the caller and does not parse untrusted input formats,
open network connections, or execute user-supplied code. The most likely
security-relevant issues are therefore memory-safety problems surfaced through
the optional CuPy backend, or dependency vulnerabilities. Reports about the
behaviour of `numpy` or `cupy` themselves should go to those projects.
