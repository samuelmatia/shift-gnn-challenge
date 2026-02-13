# Webhook Setup - Quick Start Guide

## 🎯 Goal
Trigger GitHub Actions workflow **instantly** when a form submission arrives (instead of waiting for scheduled intervals).

---

## 📋 Step-by-Step Setup (15 minutes)

### Step 1: Create GitHub Personal Access Token (5 min)

1. Go to: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Configure:
   - **Note**: `Shift-GNN Webhook`
   - **Expiration**: Your choice (90 days recommended)
   - **Scopes**: Check ✅ `repo` and ✅ `workflow`
4. Click **"Generate token"**
5. **COPY THE TOKEN** (starts with `ghp_...`) - you won't see it again!

---

### Step 2: Set Up Google Apps Script (10 min)

1. **Open your Google Sheet** (the one linked to your Google Form)

2. **Open Apps Script**:
   - Click **Extensions** → **Apps Script**

3. **Paste the code**:
   - Open `docs/GOOGLE_APPS_SCRIPT_CODE.js` from this repository
   - Copy ALL the code
   - Paste it into Apps Script editor
   - **Update CONFIG section** (lines 15-22):
     ```javascript
     const CONFIG = {
       REPO_OWNER: 'samuelmatia',  // ✅ Already correct
       REPO_NAME: 'shift-gnn-challenge',  // ✅ Already correct
       WORKFLOW_FILE: 'process_google_form_polling.yml',  // ✅ Already correct
       GITHUB_PAT: ''  // Leave empty
     };
     ```

4. **Save**: Click **File** → **Save** (or `Ctrl+S`)

5. **Store GitHub PAT**:
   - In the `setupGitHubPAT()` function (around line 120), replace `'YOUR_GITHUB_PAT_HERE'` with your actual token
   - Click **Run** → Select `setupGitHubPAT` → Click **Run** (▶️)
   - **Authorize** when prompted:
     - Click **"Review permissions"**
     - Choose your Google account
     - Click **"Advanced"** → **"Go to [Project Name] (unsafe)"**
     - Click **"Allow"**
   - Check **Execution log** (View → Logs):
     - Should see: `✅ GitHub PAT stored securely`
     - Should see: `✅ Token test successful! Setup complete.`

6. **⚠️ SECURITY**: After setup, delete or comment out the `setupGitHubPAT()` function

7. **Set up trigger**:
   - Click **Triggers** (⏰ clock icon on left)
   - Click **"+ Add Trigger"** (bottom right)
   - Configure:
     - **Function**: `onFormSubmit`
     - **Event source**: `From form`
     - **Event type**: `On form submit`
   - Click **"Save"**
   - **Authorize** again if prompted

---

### Step 3: Test (2 min)

**Option A: Test via Apps Script**
1. In Apps Script, click **Run** → Select `testWorkflowTrigger`
2. Click **Run** (▶️)
3. Check logs for: `✅ Workflow triggered successfully!`
4. Go to GitHub → **Actions** → Check if workflow ran

**Option B: Test via Form**
1. Submit a test entry via your Google Form
2. Wait 10-30 seconds
3. Go to GitHub → **Actions** → Check if workflow was triggered

---

## ✅ Verification

After setup, when someone submits the form:

1. **Form submission** → Google Sheet receives data
2. **Apps Script trigger** → `onFormSubmit()` runs automatically
3. **GitHub API call** → Workflow is triggered instantly
4. **GitHub Actions** → Processes submission and updates leaderboard
5. **Leaderboard updated** → Public leaderboard reflects new submission

**Time from submission to leaderboard update**: ~30-60 seconds (instead of waiting up to 15 minutes)

---

## 🔍 Troubleshooting

### "GitHub PAT not configured"
- Run `setupGitHubPAT()` function again
- Make sure you replaced `YOUR_GITHUB_PAT_HERE` with actual token

### "401 Unauthorized"
- Token expired or incorrect
- Regenerate token and run `setupGitHubPAT()` again

### "404 Not Found"
- Check `REPO_OWNER`, `REPO_NAME`, and `WORKFLOW_FILE` in CONFIG
- Verify workflow file exists: `.github/workflows/process_google_form_polling.yml`

### Trigger not firing
- Check trigger is set up: **Triggers** → Should see `onFormSubmit` trigger
- Verify Google Sheet is linked to Google Form
- Check execution log for errors

### Workflow not appearing in GitHub Actions
- Check GitHub Actions tab: https://github.com/samuelmatia/shift-gnn-challenge/actions
- Look for workflow runs triggered by `workflow_dispatch`

---

## 📚 Full Documentation

For detailed instructions, see: `docs/WEBHOOK_SETUP.md`

For the complete script code, see: `docs/GOOGLE_APPS_SCRIPT_CODE.js`

---

## 🎉 Result

✅ **Instant triggering** - No waiting for scheduled intervals  
✅ **More reliable** - Triggers exactly when needed  
✅ **Better UX** - Submissions processed immediately  
✅ **Backup schedule** - Still runs every 15 minutes as fallback

---

## 🔒 Security Notes

- GitHub PAT is stored securely in Script Properties (encrypted)
- Never commit tokens to code
- Delete `setupGitHubPAT()` function after setup
- Use token with minimal scopes (`repo`, `workflow` only)
- Set token expiration (90 days recommended)
