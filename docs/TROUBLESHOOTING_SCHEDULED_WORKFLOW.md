# Troubleshooting: Scheduled Workflow Not Triggering Automatically

## Common Reasons Why Scheduled Workflows Don't Trigger

### 1. ⏱️ **Activation Delay**
GitHub Actions can take **10-15 minutes** to activate a new schedule after it's committed. If you just pushed the workflow, wait a bit longer.

**Solution**: Wait 15-20 minutes and check again.

---

### 2. 📍 **Workflow Must Be on Default Branch**
Scheduled workflows **only run from the default branch** (usually `main` or `master`).

**Check**:
- Go to your repository → **Settings** → **Branches**
- Verify your default branch is `main`
- Ensure the workflow file `.github/workflows/process_google_form_polling.yml` exists on `main` branch

**Solution**: 
```bash
git checkout main
git pull origin main
# Verify the file exists
ls -la .github/workflows/process_google_form_polling.yml
```

---

### 3. 🔄 **Repository Activity Requirement**
Scheduled workflows only run if the repository has had **activity in the last 60 days** (push, PR, issue, etc.).

**Check**: Has your repository been active recently?

**Solution**: Make a small commit or create an issue to ensure activity.

---

### 4. ⚙️ **GitHub Actions Settings**
Scheduled workflows might be disabled in repository settings.

**Check**:
1. Go to **Settings** → **Actions** → **General**
2. Under "Workflow permissions", ensure:
   - ✅ "Read and write permissions" is selected (or "Read repository contents and packages permissions" with explicit `permissions:` in workflow)
   - ✅ "Allow GitHub Actions to create and approve pull requests" (if needed)
3. Under "Actions permissions":
   - ✅ "Allow all actions and reusable workflows" is selected

---

### 5. 🕐 **Cron Syntax Issue**
The cron syntax `*/5 * * * *` means "every 5 minutes". However, GitHub Actions schedules are in **UTC time**.

**Check**: Verify the cron syntax is correct:
- `*/5 * * * *` = Every 5 minutes
- `0 * * * *` = Every hour at minute 0
- `0 */2 * * *` = Every 2 hours at minute 0

**Note**: GitHub Actions may not run exactly on schedule - there can be delays of a few minutes.

---

### 6. 📊 **Check Workflow Run History**
Even if scheduled workflows don't show as "in progress", they might have run and completed quickly.

**Check**:
1. Go to **Actions** tab
2. Click on **"Process Google Form Submissions (Polling)"**
3. Look at the run history - do you see runs every 5 minutes?
4. Check if runs completed successfully (green checkmark) or failed (red X)

---

### 7. 🔍 **Verify Workflow File Location**
The workflow file must be in the correct location.

**Check**:
- File path: `.github/workflows/process_google_form_polling.yml`
- File exists on `main` branch
- File syntax is valid YAML

**Solution**:
```bash
# Verify file exists
git ls-files .github/workflows/process_google_form_polling.yml

# Check file content
cat .github/workflows/process_google_form_polling.yml
```

---

## Diagnostic Steps

### Step 1: Verify Workflow File
```bash
cd /path/to/repo
git checkout main
git pull origin main
cat .github/workflows/process_google_form_polling.yml
```

### Step 2: Check Recent Workflow Runs
1. Go to GitHub → **Actions** tab
2. Look for "Process Google Form Submissions (Polling)"
3. Check if there are any recent runs (even if they failed)

### Step 3: Test Manual Trigger
1. Go to **Actions** → **Process Google Form Submissions (Polling)**
2. Click **"Run workflow"** → **"Run workflow"**
3. If manual trigger works but scheduled doesn't, it's likely a scheduling issue

### Step 4: Check Repository Activity
- Make a small commit to ensure repository is active
- Or create an issue/PR

### Step 5: Wait and Monitor
- Wait 15-20 minutes after any changes
- Check Actions tab periodically
- Scheduled workflows may have slight delays

---

## Alternative: Use GitHub Actions with Shorter Intervals

If scheduled workflows are unreliable, consider:

1. **Increase interval** to 10-15 minutes (more reliable)
2. **Use webhook-based approach** (Google Apps Script → GitHub webhook) for instant updates
3. **Manual trigger** when needed (less automated but more reliable)

---

## Quick Fix: Test with Longer Interval

If you want to test if scheduling works at all, try changing to hourly:

```yaml
schedule:
  - cron: '0 * * * *'  # Every hour at minute 0
```

This is more reliable and easier to verify.

---

## Still Not Working?

If none of the above works:

1. **Check GitHub Status**: https://www.githubstatus.com/ - ensure Actions is operational
2. **Repository Type**: Ensure it's not a fork (scheduled workflows don't run on forks by default)
3. **Account Limits**: Free accounts have limits on Actions minutes
4. **Contact GitHub Support**: If everything else checks out, there might be an account-specific issue

---

## Expected Behavior

Once working correctly, you should see:
- Workflow runs every 5 minutes (with possible delays)
- Runs appear in Actions tab with timestamps
- Each run processes new submissions and updates leaderboard
- Runs complete successfully (green checkmark)

---

## Verification Checklist

- [ ] Workflow file exists on `main` branch
- [ ] Cron syntax is correct (`*/5 * * * *`)
- [ ] Repository has been active in last 60 days
- [ ] GitHub Actions settings allow scheduled workflows
- [ ] Manual trigger works (test via "Run workflow")
- [ ] Waited 15-20 minutes after committing workflow
- [ ] Checked Actions tab for any runs (even failed ones)
