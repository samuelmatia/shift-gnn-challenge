# Fix: Scheduled Workflow Not Triggering

## Problem
GitHub Actions scheduled workflows (`schedule:`) can be unreliable, especially for frequent intervals (every 5 minutes). They may not trigger consistently due to:
- GitHub Actions infrastructure delays
- Repository activity requirements
- Free tier limitations
- Cron interpretation delays

## Solution Options

### Option 1: Increase Interval (Quick Fix)
Change from 5 minutes to 15-30 minutes for more reliability:

```yaml
schedule:
  - cron: '*/15 * * * *'  # Every 15 minutes (more reliable)
```

### Option 2: Use Google Apps Script Webhook (Recommended)
Trigger workflow instantly when a new form submission arrives.

**Setup Steps:**

1. **Create GitHub Personal Access Token (PAT)**:
   - GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with `repo` and `workflow` scopes
   - Save token securely

2. **Add PAT as GitHub Secret**:
   - Repository → Settings → Secrets → Actions
   - Add secret: `GITHUB_PAT` (your personal access token)

3. **Create Google Apps Script**:
   - Open your Google Sheet
   - Extensions → Apps Script
   - Paste this code:

```javascript
function onFormSubmit(e) {
  // Trigger GitHub Actions workflow via webhook
  const githubToken = 'YOUR_GITHUB_PAT'; // Or use PropertiesService for security
  const repoOwner = 'samuelmatia';
  const repoName = 'shift-gnn-challenge';
  const workflowId = 'process_google_form_polling.yml';
  
  const url = `https://api.github.com/repos/${repoOwner}/${repoName}/actions/workflows/${workflowId}/dispatches`;
  
  const options = {
    'method': 'post',
    'headers': {
      'Authorization': `token ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json'
    },
    'payload': JSON.stringify({
      'ref': 'main'
    })
  };
  
  try {
    const response = UrlFetchApp.fetch(url, options);
    Logger.log('Workflow triggered: ' + response.getResponseCode());
  } catch (error) {
    Logger.log('Error triggering workflow: ' + error);
  }
}

// Install trigger: Edit → Current project's triggers → Add trigger
// Event: On form submit
```

4. **Set up Trigger**:
   - In Apps Script: Edit → Current project's triggers
   - Add trigger: `onFormSubmit`, Event: "On form submit"

### Option 3: Use GitHub Actions with Longer Interval + Manual Trigger
Keep scheduled workflow but use longer interval (hourly) and rely on manual triggers when needed.

### Option 4: External Cron Service
Use an external service (like cron-job.org) to call GitHub API every 5 minutes to trigger workflow.

---

## Immediate Fix: Test with Hourly Schedule

Let's first verify if scheduling works at all with a longer interval:

```yaml
schedule:
  - cron: '0 * * * *'  # Every hour at minute 0
```

This is more reliable and easier to verify.

---

## Verification Steps

1. **Check if workflow file is on main branch**:
   ```bash
   git checkout main
   git pull origin main
   cat .github/workflows/process_google_form_polling.yml
   ```

2. **Verify workflow syntax**:
   - Go to Actions → Your workflow
   - Check if there are any syntax errors

3. **Check repository activity**:
   - Make a small commit to ensure repository is active
   - Scheduled workflows require recent activity

4. **Test manual trigger**:
   - Actions → Process Google Form Submissions (Polling) → Run workflow
   - If manual works but scheduled doesn't, it's a scheduling issue

5. **Check GitHub Actions status**:
   - https://www.githubstatus.com/
   - Ensure Actions service is operational

---

## Recommended Approach

For a production system, I recommend **Option 2 (Google Apps Script Webhook)** because:
- ✅ Instant triggering (no waiting)
- ✅ More reliable than scheduled workflows
- ✅ Only runs when needed (when form is submitted)
- ✅ No dependency on GitHub Actions scheduling

The scheduled workflow can remain as a backup (with longer interval like hourly) in case the webhook fails.
