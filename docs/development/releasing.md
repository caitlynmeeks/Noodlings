# Releasing

**How to create and publish releases**

---

## Overview

Releases follow semantic versioning (semver) with pre-release suffixes for alpha/beta stages.

**Current version:** See `VERSION` file in repository root.

---

## Version Format

```
MAJOR.MINOR.PATCH[-PRERELEASE]

Examples:
  0.1.0-alpha    First alpha
  0.1.1-alpha    Bug fix in alpha
  0.2.0-alpha    New feature in alpha
  0.9.0-beta     Feature complete, testing
  0.9.1-rc.1     Release candidate 1
  1.0.0          First stable release
```

### When to Bump

| Change Type | Version Bump | Example |
|-------------|--------------|---------|
| Bug fix | PATCH | 0.1.0 -> 0.1.1 |
| New feature (backward compatible) | MINOR | 0.1.1 -> 0.2.0 |
| Breaking change | MAJOR | 0.2.0 -> 1.0.0 |
| Ready for beta | Suffix | 0.5.0-alpha -> 0.5.0-beta |
| Release candidate | Suffix | 0.9.0-beta -> 0.9.0-rc.1 |

---

## Release Checklist

### 1. Ensure Tests Pass

```bash
cd applications/noodlestudio
PYTHONPATH=.:../.. pytest -v
```

All tests must pass. No exceptions.

### 2. Update VERSION File

Edit `VERSION` in repository root:

```bash
echo "0.2.0-alpha" > VERSION
```

### 3. Update __init__.py

Edit `applications/noodlestudio/noodlestudio/__init__.py`:

```python
__version__ = "0.2.0-alpha"
```

### 4. Update CHANGELOG.md

Move items from `[Unreleased]` to new version section:

```markdown
## [Unreleased]

(empty or new unreleased items)

---

## [0.2.0-alpha] - 2026-01-15

### Added
- New feature X
- New feature Y

### Fixed
- Bug in Z

### Changed
- Improved performance of W
```

### 5. Commit the Release

```bash
git add VERSION CHANGELOG.md applications/noodlestudio/noodlestudio/__init__.py
git commit -m "Release 0.2.0-alpha"
```

### 6. Tag the Release

```bash
git tag -a v0.2.0-alpha -m "Version 0.2.0-alpha"
```

### 7. Push

```bash
git push origin main
git push origin v0.2.0-alpha
```

### 8. Create GitHub Release (Optional)

1. Go to repository on GitHub
2. Click "Releases"
3. Click "Create a new release"
4. Select the tag
5. Copy changelog section as release notes
6. Publish

---

## Version Locations

Keep these in sync:

| Location | Format |
|----------|--------|
| `VERSION` | `0.2.0-alpha` (no quotes, single line) |
| `__init__.py` | `__version__ = "0.2.0-alpha"` |
| `CHANGELOG.md` | `## [0.2.0-alpha] - YYYY-MM-DD` |
| Git tag | `v0.2.0-alpha` (note the `v` prefix) |

---

## Changelog Format

Follow [Keep a Changelog](https://keepachangelog.com/):

### Section Types

- **Added** - New features
- **Changed** - Changes to existing functionality
- **Deprecated** - Features to be removed
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security fixes

### Good Changelog Entries

```markdown
### Added
- Gaussian viewer panel with orbit controls and bone visualization
- Pre-commit hook for automated testing

### Fixed
- Inspector now preserves selection when switching tabs
- Stage View no longer crashes when deleting last zone
```

### Bad Changelog Entries

```markdown
### Added
- Stuff
- Fixed things
- Updated code
```

---

## Pre-release vs Stable

### Alpha (-alpha)

- Active development
- Features may change or break
- Not recommended for production

### Beta (-beta)

- Feature complete
- Stabilization phase
- API may still change

### Release Candidate (-rc.N)

- Feature frozen
- Only critical bug fixes
- Preparing for stable release

### Stable (no suffix)

- Production ready
- Backward compatibility guaranteed within major version
- Security fixes backported

---

## Rollback

If a release has critical bugs:

```bash
# Revert to previous version
git revert HEAD
git tag -d v0.2.0-alpha
git push origin :refs/tags/v0.2.0-alpha
git push origin main
```

Then fix the issue and release as a new patch version.

---

## See Also

- [CI/CD](ci-cd.md) - Automated testing
- [Testing](../noodlestudio/testing.md) - Test suite
