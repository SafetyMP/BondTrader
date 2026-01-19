#!/bin/bash
# Git History Cleanup Script - Remove API Keys from Git History
# WARNING: This rewrites git history and requires force pushing

set -e  # Exit on error

REPO_PATH="/Users/sagehart/Downloads/BondTrader"
cd "$REPO_PATH"

echo "=========================================="
echo "Git History Cleanup Script"
echo "=========================================="
echo ""
echo "⚠️  WARNING: This script will:"
echo "   1. Rewrite ALL git commit history"
echo "   2. Change ALL commit hashes"
echo "   3. Require force pushing to GitHub"
echo "   4. Require collaborators to re-clone"
echo ""
echo "⚠️  Make sure you:"
echo "   - Have rotated the API keys at source (FRED/FINRA)"
echo "   - Have backed up your repository"
echo "   - Have coordinated with collaborators"
echo ""
read -p "Press Enter to continue or Ctrl+C to cancel..."

# Step 1: Create backup
echo ""
echo "Step 1: Creating backup..."
cd ..
if [ ! -d "BondTrader-backup-$(date +%Y%m%d-%H%M%S)" ]; then
    BACKUP_DIR="BondTrader-backup-$(date +%Y%m%d-%H%M%S)"
    echo "   Creating backup in: $BACKUP_DIR"
    git clone BondTrader "$BACKUP_DIR"
    echo "   ✅ Backup created"
else
    echo "   ⚠️  Backup directory already exists, skipping backup"
fi
cd "$REPO_PATH"

# Step 2: Check/Install git-filter-repo
echo ""
echo "Step 2: Checking for git-filter-repo..."
if ! command -v git-filter-repo &> /dev/null; then
    echo "   git-filter-repo not found. Installing..."
    if command -v pip3 &> /dev/null; then
        pip3 install git-filter-repo
    elif command -v pip &> /dev/null; then
        pip install git-filter-repo
    else
        echo "   ❌ ERROR: pip not found. Please install git-filter-repo manually:"
        echo "      brew install git-filter-repo"
        echo "      OR: pip install git-filter-repo"
        exit 1
    fi
    echo "   ✅ git-filter-repo installed"
else
    echo "   ✅ git-filter-repo found"
fi

# Step 3: Create replacements file
echo ""
echo "Step 3: Creating replacements file..."
cat > /tmp/git-cleanup-replacements.txt << 'EOF'
58bfd66ff30c430fdc4a965ad7ac9dbe==>your_fred_api_key_here
ec38ead419a84e30acc2==>your_finra_api_key_here
zekfus-gutZap-5nohvye==>your_finra_password_here
EOF
echo "   ✅ Replacements file created at /tmp/git-cleanup-replacements.txt"

# Step 4: Verify current state
echo ""
echo "Step 4: Checking current state..."
if git log --all -p | grep -qE "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"; then
    echo "   ⚠️  Keys found in git history (this is expected)"
else
    echo "   ✅ No keys found in history (already cleaned?)"
    read -p "   Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "   Aborted."
        exit 0
    fi
fi

# Step 5: Run git-filter-repo
echo ""
echo "Step 5: Running git-filter-repo..."
echo "   This may take a few minutes..."
git filter-repo --replace-text /tmp/git-cleanup-replacements.txt
echo "   ✅ git-filter-repo completed"

# Step 6: Verify cleanup
echo ""
echo "Step 6: Verifying cleanup..."
if git log --all -p | grep -qE "58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye"; then
    echo "   ❌ ERROR: Keys still found in history!"
    echo "   Please check manually: git log --all -p | grep -E '58bfd66ff30c430fdc4a965ad7ac9dbe|ec38ead419a84e30acc2|zekfus-gutZap-5nohvye'"
    exit 1
else
    echo "   ✅ Success: Keys removed from history"
fi

# Step 7: Show summary
echo ""
echo "=========================================="
echo "Cleanup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Review the changes:"
echo "   git log --oneline"
echo ""
echo "2. Force push to GitHub (⚠️  DESTRUCTIVE):"
echo "   git push origin --force --all"
echo "   git push origin --force --tags"
echo ""
echo "3. Notify collaborators:"
echo "   They need to delete and re-clone the repository:"
echo "   rm -rf BondTrader"
echo "   git clone https://github.com/SafetyMP/BondTrader.git"
echo ""
echo "4. ⚠️  IMPORTANT: Rotate the actual API keys!"
echo "   - FRED: https://fred.stlouisfed.org/docs/api/api_key.html"
echo "   - FINRA: https://www.finra.org/finra-data/browse-catalog"
echo ""
echo "5. Check GitHub secret scanning alerts"
echo ""
