# Security Policy

## Supported versions

The latest release on PyPI is supported. Older releases are not.

## Reporting a vulnerability

Email **security@arbez.org**, or use GitHub's private vulnerability
reporting at
<https://github.com/arbez-org/arbez-sdk-python/security/advisories/new>.
Either way, include:

- Affected version(s)
- Minimal reproducer
- Impact assessment (RCE / data leak / DoS / etc.)
- Suggested fix if you have one

We acknowledge within 5 business days. Fix target: 90 days. CVE coordination via GitHub Security Advisories.

**Please do not file vulnerability reports as public GitHub issues.**

## Scope

- **In scope:** `arbez/*` source code, the bundled `arbez_yolox_s.onnx` model (model-extraction / adversarial-perturbation issues), published wheels
- **Out of scope:** upstream dependency vulnerabilities (report to those projects directly); vulnerabilities introduced by user-supplied weights via `bring-your-own-weights`
