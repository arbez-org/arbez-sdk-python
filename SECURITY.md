# Security Policy

## Supported versions

The latest `0.0.x` release on test.pypi / PyPI is supported. Older 0.0.x releases are not.

## Reporting a vulnerability

Email **security@arbez.org** with:

- Affected version(s)
- Minimal reproducer
- Impact assessment (RCE / data leak / DoS / etc.)
- Suggested fix if you have one

We acknowledge within 5 business days. Fix target: 90 days. CVE coordination via GitHub Security Advisories.

**Please do not file vulnerability reports as public GitHub issues.**

## Scope

- **In scope:** `arbez/*` source code, the bundled `arbez_yolox_s.onnx` model (model-extraction / adversarial-perturbation issues), published wheels
- **Out of scope:** upstream dependency vulnerabilities (report to those projects directly); vulnerabilities introduced by user-supplied weights via `bring-your-own-weights`
