# Git History Cleanup Guide: Removing API Keys

## ⚠️ CRITICAL WARNINGS

**Before proceeding, understand:**
1. This **rewrites git history** - all commit hashes will change
2. Anyone who has cloned the repo will need to re-clone or reset their local copy
3. This requires **force pushing** to GitHub
4. If others are working on this repo, coordinate with them first
5. **Always backup your repository** before starting

## Method 1: Using git-filter-repo (Recommended) ⭐

This is the modern, recommended tool for rewriting git history.

### Step 1: Install git-filter-repo

```bash
# macOS
brew install git-filter-repo

# Or using pip
pip install git-filter-repo

# Verify installation
git filter-repo --version
```

### Step 2: Create a backup

```bash
cd /Users/sagehart/Downloads/BondTrader
cd ..
git clone BondTrader BondTrader-backup
```

### Step 3: Remove the API keys from history

```bash
cd /Users/sagehart/Downloads/BondTrader

# Create a replacements file
cat > /tmp/replacements.txt << 'EOF'
58bfd66ff30c430fdc4a965ad7ac9dbe==>your_fred_api_key_here
ec38ead419a84e30acc2==>your_finra_api_key_here
zekfus-gutZap-5nohvye==>your_finra_password_here
EOF

# Run git-filter-repo to replace the keys
git filter-repo --replace-text /tmp/replacements.txt

# Verify the keys are gone
git log --all -p | grep -E "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"
# Should return nothing if successful
```

### Step 4: Force push to GitHub

```bash
# WARNING: This rewrites history on GitHub!
git push origin --force --all
git push origin --force --tags
```

### Step 5: Notify collaborators

Anyone who has cloned the repo needs to:
```bash
# Option A: Delete and re-clone (easiest)
rm -rf BondTrader
git clone https://github.com/SafetyMP/BondTrader.git

# Option B: Reset their local copy
git fetch origin
git reset --hard origin/main
```

---

## Method 2: Using BFG Repo-Cleaner (Alternative)

BFG is another popular tool for cleaning git history.

### Step 1: Install BFG

```bash
# macOS
brew install bfg

# Or download from: https://rtyley.github.io/bfg-repo-cleaner/
```

### Step 2: Create replacements file

```bash
cat > /tmp/replacements.txt << 'EOF'
58bfd66ff30c430fdc4a965ad7ac9dbe==>your_fred_api_key_here
ec38ead419a84e30acc2==>your_finra_api_key_here
zekfus-gutZap-5nohvye==>your_finra_password_here
EOF
```

### Step 3: Clean the repository

```bash
cd /Users/sagehart/Downloads/BondTrader

# Clone a fresh copy (BFG needs a bare repo)
cd ..
git clone --mirror BondTrader BondTrader.git

# Run BFG
cd BondTrader.git
bfg --replace-text /tmp/replacements.txt

# Clean up and push
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Push back
git push --force

# Update your working copy
cd ../BondTrader
git fetch origin
git reset --hard origin/main
```

---

## Method 3: Manual git filter-branch (Not Recommended)

This is the older method. Use only if you can't install the other tools.

```bash
cd /Users/sagehart/Downloads/BondTrader

# Replace keys in all commits
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch .env.example' \
  --prune-empty --tag-name-filter cat -- --all

# Then manually edit .env.example in each commit (complex!)
# This method is error-prone - use Method 1 or 2 instead
```

---

## Verification Steps

After cleanup, verify the keys are gone:

```bash
# Search entire git history for the keys
git log --all --full-history -p | grep -E "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"

# Should return nothing if successful

# Check specific file history
git log --all --full-history -- .env.example | head -20

# Verify current file doesn't have keys
cat .env.example | grep -E "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"
# Should return nothing
```

---

## Quick Script (git-filter-repo method)

Here's a complete script you can run:

```bash
#!/bin/bash
set -e

REPO_PATH="/Users/sagehart/Downloads/BondTrader"
cd "$REPO_PATH"

echo "⚠️  WARNING: This will rewrite git history!"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Backup
echo "Creating backup..."
cd ..
if [ ! -d "BondTrader-backup" ]; then
    git clone BondTrader BondTrader-backup
fi
cd "$REPO_PATH"

# Create replacements file
cat > /tmp/replacements.txt << 'EOF'
58bfd66ff30c430fdc4a965ad7ac9dbe==>your_fred_api_key_here
ec38ead419a84e30acc2==>your_finra_api_key_here
zekfus-gutZap-5nohvye==>your_finra_password_here
EOF

# Check if git-filter-repo is installed
if ! command -v git-filter-repo &> /dev/null; then
    echo "Installing git-filter-repo..."
    pip install git-filter-repo
fi

# Run cleanup
echo "Cleaning git history..."
git filter-repo --replace-text /tmp/replacements.txt

# Verify
echo "Verifying cleanup..."
if git log --all -p | grep -qE "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"; then
    echo "❌ ERROR: Keys still found in history!"
    exit 1
else
    echo "✅ Success: Keys removed from history"
fi

echo ""
echo "Next steps:"
echo "1. Review the changes: git log --oneline"
echo "2. Force push to GitHub: git push origin --force --all"
echo "3. Notify collaborators to re-clone the repository"
echo ""
echo "⚠️  Remember to rotate the actual API keys!"
```

Save this as `cleanup-history.sh`, make it executable (`chmod +x cleanup-history.sh`), and run it.

---

## Important Notes

1. **GitHub Secret Scanning**: Even after removing from history, GitHub may have already scanned and detected these secrets. Check GitHub's security alerts.

2. **API Key Rotation**: Removing from git history doesn't invalidate the keys. You **MUST** rotate them at the source (FRED/FINRA).

3. **Backup First**: Always backup before rewriting history.

4. **Coordinate**: If others use this repo, coordinate the cleanup with them.

5. **Force Push**: After cleanup, you'll need to force push. This is destructive - be certain before proceeding.

---

## After Cleanup Checklist

- [ ] Backup created
- [ ] Keys removed from git history (verified)
- [ ] Force pushed to GitHub
- [ ] Collaborators notified
- [ ] API keys rotated at source (FRED/FINRA)
- [ ] GitHub secret scanning alerts checked
- [ ] Pre-commit hooks installed to prevent future incidents

---

## Need Help?

If you encounter issues:
1. Check the backup repository
2. Review git-filter-repo documentation: https://github.com/newren/git-filter-repo
3. Consider using GitHub's built-in secret scanning and alerts
