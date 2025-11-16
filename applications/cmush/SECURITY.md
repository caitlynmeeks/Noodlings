# Security - Keeping API Keys Safe

**IMPORTANT**: Never commit API keys to GitHub!

---

## ✅ What's Protected

Your `.gitignore` now includes:
- `config.yaml` - Contains your actual API key
- `.env` - Environment variables with secrets
- `*.key` - Any key files
- `*.secret` - Any secret files

**These files are NEVER pushed to GitHub!**

---

## 🔒 How to Use Secrets Safely

### **Step 1**: Create `.env` file
```bash
cd /Users/thistlequell/git/noodlings_clean/applications/cmush
cp .env.example .env
```

### **Step 2**: Add your key to `.env`
```bash
# Edit .env
OPENROUTER_API_KEY=sk-or-v1-YOUR_ACTUAL_KEY_HERE
```

### **Step 3**: Config reads from environment
```yaml
# config.yaml
openrouter:
  api_key: ${OPENROUTER_API_KEY}  # ← Reads from .env
```

### **Step 4**: Never commit config.yaml or .env
```bash
# .gitignore already has these!
config.yaml
.env
```

---

## 🚨 Your Key Was Exposed!

**What happened**: You pasted your OpenRouter key in our conversation

**Risk**: MODERATE (conversation is private, but better safe than sorry)

**Fix**: Rotate your key NOW

### **How to Rotate**:

1. **Go to**: https://openrouter.ai/keys
2. **Delete old key**: `sk-or-v1-956bf0...` (the one you pasted)
3. **Create new key**: Click "Create Key"
4. **Copy new key**: Save it somewhere safe
5. **Update `.env`**: Paste new key
6. **Restart noodleMUSH**: New key is now used

**Cost**: $0 (rotating is free!)

---

## ✅ Safe Configuration

**DO** ✅:
- Use `.env` files for secrets
- Keep `.env` in `.gitignore`
- Use `config.example.yaml` (with placeholder keys) for GitHub
- Rotate keys if exposed

**DON'T** ❌:
- Commit `config.yaml` with real keys
- Paste keys in chat/emails
- Share keys in screenshots
- Check keys into version control

---

## 📋 Checklist for Steve Demo

Before demo:
- [ ] Rotate OpenRouter key (get fresh one)
- [ ] Update `.env` with new key
- [ ] Test: `config.yaml` reads from `.env` correctly
- [ ] Verify: `.env` is NOT in git (`git status` shouldn't show it)
- [ ] Add $5 to OpenRouter credits

---

**Your secrets are now safe!** 🔒✨

Just rotate that key and you're good to go! 🚀
