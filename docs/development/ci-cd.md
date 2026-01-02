# CI/CD Pipeline

**Continuous Integration and Deployment for Noodlings**

---

## Overview

The CI/CD pipeline consists of two layers:

1. **Local** - Pre-commit hook runs tests before commits
2. **Remote** - GitHub Actions runs tests on push/PR

Both use the same test suite. If tests pass locally, they should pass remotely.

---

## GitHub Actions

### Workflow File

`.github/workflows/ci.yml`

### Triggers

- Push to `main` or `master` branch
- Pull requests targeting `main` or `master`

### What It Does

1. Checks out code
2. Sets up Python 3.12
3. Installs dependencies
4. Runs pytest on NoodleStudio tests

### Viewing Results

- Go to repository on GitHub
- Click "Actions" tab
- Green checkmark = tests passed
- Red X = tests failed (click for details)

### Build Status Badge

Add to README:

```markdown
![CI](https://github.com/noodlings-ai/noodlings/actions/workflows/ci.yml/badge.svg)
```

---

## Pre-commit Hook

### Location

`.git/hooks/pre-commit`

### What It Does

1. Runs pytest on NoodleStudio tests
2. Blocks commit if tests fail
3. Shows colored output (green = pass, red = fail)

### Bypassing

For work-in-progress commits where tests are expected to fail:

```bash
git commit --no-verify -m "WIP: description"
```

Use sparingly. Tests should pass before pushing.

### Reinstalling

If the hook is missing (fresh clone, etc.):

```bash
# The hook lives in .git/hooks/ which isn't tracked by git
# Copy from docs or recreate:

cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
set -e
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/applications/noodlestudio"
PYTHONPATH=".:../.." TOKENIZERS_PARALLELISM=false \
  "$REPO_ROOT/venv/bin/python" -m pytest tests/ --tb=short -q
EOF

chmod +x .git/hooks/pre-commit
```

---

## Test Environment

### Local Requirements

- Python 3.10+ (3.12 recommended)
- Virtual environment at `venv/`
- Dependencies from `requirements.txt`

### CI Environment

- macOS latest (for Apple Silicon compatibility)
- Python 3.12
- Qt platform: offscreen (headless)

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `PYTHONPATH` | Include project root for imports |
| `TOKENIZERS_PARALLELISM` | Suppress HuggingFace warning |
| `QT_QPA_PLATFORM` | Run Qt headless in CI |

---

## Troubleshooting

### Tests Pass Locally, Fail in CI

1. Check Python version (CI uses 3.12)
2. Check for missing dependencies in requirements.txt
3. Check for hardcoded paths (use relative paths)
4. Check for GPU-dependent code (CI has no GPU)

### Pre-commit Hook Not Running

```bash
# Check hook exists and is executable
ls -la .git/hooks/pre-commit

# Make executable if needed
chmod +x .git/hooks/pre-commit
```

### CI Takes Too Long

- Tests should complete in under 5 minutes
- If slow, check for:
  - Unnecessary imports at module level
  - Tests that wait for timeouts
  - Missing `@pytest.mark.slow` markers

---

## Future Enhancements

When needed (not now):

- [ ] Linting with ruff
- [ ] Type checking with pyright
- [ ] Coverage reporting
- [ ] Automated releases on tag push
- [ ] Multi-platform testing (Linux, Windows)

---

## See Also

- [Testing Guide](../noodlestudio/testing.md) - Writing and running tests
- [Releasing Guide](releasing.md) - Release process
