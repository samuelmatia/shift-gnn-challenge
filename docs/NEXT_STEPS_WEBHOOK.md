# Next Steps: Finalize Webhook Setup

## ✅ Current Status
- GitHub Personal Access Token configured ✅
- Token test successful ✅
- Workflow triggered successfully ✅

---

## Step 1: Secure Your Token (IMPORTANT - 2 minutes)

**⚠️ SECURITY**: Remove or comment out the `setupGitHubPAT()` function to prevent accidental token exposure.

### In Google Apps Script:

1. Find the `setupGitHubPAT()` function (around line 142)
2. **Option A**: Delete the entire function
3. **Option B**: Comment it out:
   ```javascript
   /*
   function setupGitHubPAT() {
     // ... entire function ...
   }
   */
   ```

**Why?** The token is now safely stored in Script Properties. The function is no longer needed and could expose your token if someone accesses the script.

---

## Step 2: Set Up Automatic Trigger (3 minutes)

Configure the script to run automatically when a form is submitted.

### In Google Apps Script:

1. Click **Triggers** (⏰ clock icon on the left sidebar)
2. Click **"+ Add Trigger"** (bottom right)
3. Configure the trigger:
   - **Choose which function to run**: `onFormSubmit`
   - **Select event source**: `From form`
   - **Select event type**: `On form submit`
   - **Failure notification settings**: Choose your preference (e.g., "Notify me immediately")
4. Click **"Save"**
5. **Authorize** if prompted:
   - Click **"Review permissions"**
   - Choose your Google account
   - Click **"Advanced"** → **"Go to [Project Name] (unsafe)"**
   - Click **"Allow"**

### Verify Trigger:
- You should see a trigger listed: `onFormSubmit` → `From form` → `On form submit`
- Status should be: ✅ (green checkmark)

---

## Step 3: Test with Real Form Submission (5 minutes)

### Test the Complete Flow:

1. **Submit a test entry** via your Google Form:
   - Fill out all required fields
   - Upload a CSV file
   - Submit the form

2. **Check Google Apps Script Logs** (within 10-30 seconds):
   - Go to Apps Script editor
   - Click **View** → **Logs** (or `Ctrl+Enter` / `Cmd+Enter`)
   - You should see:
     ```
     📝 Form submission detected
     🚀 Triggering GitHub Actions workflow...
     ✅ Workflow triggered successfully!
     ```

3. **Check GitHub Actions** (within 30-60 seconds):
   - Go to: https://github.com/samuelmatia/shift-gnn-challenge/actions
   - Click on **"Process Google Form Submissions (Polling)"**
   - You should see a new workflow run
   - The run should show:
     - Triggered by: `workflow_dispatch` or `repository_dispatch`
     - Status: Running → Succeeded (green checkmark)

4. **Verify Leaderboard Update**:
   - Wait for workflow to complete (~30-60 seconds)
   - Check your leaderboard: https://samuelmatia.github.io/shift-gnn-challenge/leaderboard.html
   - Or check `leaderboard.json` in your repository
   - New submission should appear on the leaderboard

---

## Step 4: Monitor and Verify (Ongoing)

### What to Monitor:

1. **Form Submissions**:
   - Each time someone submits the form, check:
     - Apps Script logs show successful trigger
     - GitHub Actions shows new workflow run
     - Leaderboard updates within 1-2 minutes

2. **Error Handling**:
   - If workflow fails, check GitHub Actions logs
   - If trigger doesn't fire, check Apps Script execution log
   - Common issues:
     - Token expired → Regenerate and update
     - Form not linked to Sheet → Verify Sheet is connected to Form
     - Trigger not set up → Re-add trigger

---

## Step 5: Optional - Adjust Scheduled Workflow (Optional)

Since webhook is now your primary method, you can:

### Option A: Keep Scheduled as Backup (Recommended)
- Keep the 15-minute schedule as a safety net
- If webhook fails, scheduled run will catch it

### Option B: Increase Scheduled Interval
- Change to hourly or daily (less frequent)
- Edit `.github/workflows/process_google_form_polling.yml`:
  ```yaml
  schedule:
    - cron: '0 * * * *'  # Every hour
    # or
    - cron: '0 0 * * *'  # Daily at midnight
  ```

### Option C: Remove Scheduled Workflow
- If webhook is 100% reliable, you can remove the schedule
- But keeping it as backup is recommended

---

## Troubleshooting

### Trigger Not Firing
- ✅ Check trigger is set up: **Triggers** → Should see `onFormSubmit` trigger
- ✅ Verify Google Sheet is linked to Google Form
- ✅ Check execution log for errors

### Workflow Not Appearing in GitHub Actions
- ✅ Check GitHub Actions tab
- ✅ Verify token still has correct scopes
- ✅ Check if workflow file exists: `.github/workflows/process_google_form_polling.yml`

### Leaderboard Not Updating
- ✅ Check workflow completed successfully (green checkmark)
- ✅ Check workflow logs for errors
- ✅ Verify `process_google_form_submissions.py` ran successfully
- ✅ Check if submission was already processed (duplicate prevention)

---

## Success Indicators

You'll know everything is working when:

✅ Form submission → Apps Script log shows "Workflow triggered successfully"  
✅ GitHub Actions shows new workflow run within 30 seconds  
✅ Workflow completes successfully (green checkmark)  
✅ Leaderboard updates with new submission within 1-2 minutes  
✅ No manual intervention needed

---

## Final Checklist

- [ ] Token stored securely (setupGitHubPAT function removed/commented)
- [ ] Trigger configured (`onFormSubmit` → `On form submit`)
- [ ] Test submission successful
- [ ] Workflow appears in GitHub Actions
- [ ] Leaderboard updates correctly
- [ ] Monitoring set up (optional: email notifications)

---

## You're Done! 🎉

Your webhook is now fully configured and will:
- ✅ Trigger instantly when form is submitted
- ✅ Process submissions automatically
- ✅ Update leaderboard within 1-2 minutes
- ✅ Work reliably without manual intervention

The system is now production-ready!
