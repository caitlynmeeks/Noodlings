# Development Guide

**Infrastructure and workflows for maintaining NoodleStudio and NoodleMUSH**

---

## Philosophy

This project is maintained by a small team (currently one person). The development infrastructure is designed around one principle:

**Design for forgiveness, not discipline.**

Systems should catch mistakes automatically, not require perfect memory or meticulous attention.

---

## Documentation Map

| Document | Purpose |
|----------|---------|
| [CI/CD](ci-cd.md) | Continuous integration, GitHub Actions |
| [Releasing](releasing.md) | Version bumps, changelog, release process |
| [Bug Reporting](bug-reporting.md) | Crash detection, issue submission |
| [Testing](../noodlestudio/testing.md) | Running and writing tests |

---

## Quick Reference

### Before Coding

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest --tb=short
```

### Before Committing

Tests run automatically via pre-commit hook. To bypass (for WIP):

```bash
git commit --no-verify -m "WIP: description"
```

### Before Releasing

See [Releasing Guide](releasing.md).

---

## Key Files

| File | Purpose |
|------|---------|
| `VERSION` | Canonical version number |
| `CHANGELOG.md` | Human-readable release history |
| `.github/workflows/ci.yml` | GitHub Actions workflow |
| `.git/hooks/pre-commit` | Local test runner |
| `pytest.ini` | Test configuration |
| `tests/conftest.py` | Shared test fixtures |

---

## Safety Nets

The following systems catch mistakes automatically:

1. **Pre-commit hook** - Runs tests before every commit
2. **GitHub Actions** - Runs tests on every push to main
3. **Claude reminders** - Prompts to run tests during development

If all three fail, something is very wrong.

---

## See Also

- [Architecture](../architecture.md) - System design
- [NoodleStudio Overview](../noodlestudio/overview.md) - IDE introduction
