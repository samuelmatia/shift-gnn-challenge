# Webhook Setup: Instant Workflow Triggering via Google Apps Script

## Overview
This setup triggers the GitHub Actions workflow **instantly** when a new submission arrives in your Google Form, eliminating the need to wait for scheduled intervals.

---

## Step 1: Create GitHub Personal Access Token (PAT)

1. Go to GitHub → Click your profile picture → **Settings**
2. Scroll down → **Developer settings** → **Personal access tokens** → **Tokens (classic)**
3. Click **"Generate new token"** → **"Generate new token (classic)"**
4. Configure the token:
   - **Note**: `Shift-GNN Challenge Workflow Trigger`
   - **Expiration**: Choose your preference (90 days, 1 year, or no expiration)
   - **Scopes**: Check these boxes:
     - ✅ `repo` (Full control of private repositories)
     - ✅ `workflow` (Update GitHub Action workflows)
5. Click **"Generate token"**
6. **⚠️ IMPORTANT**: Copy the token immediately (you won't be able to see it again!)
   - It looks like: `ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

---

## Step 2: Add PAT as GitHub Secret

1. Go to your repository: `https://github.com/samuelmatia/shift-gnn-challenge`
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **"New repository secret"**
4. Configure:
   - **Name**: `GITHUB_PAT`
   - **Secret**: Paste your Personal Access Token
5. Click **"Add secret"**

---

## Step 3: Create Google Apps Script

1. Open your Google Sheet (the one linked to your Google Form)
2. Click **Extensions** → **Apps Script**
3. Delete any existing code
4. Paste the following code:

```javascript
/**
 * Trigger GitHub Actions workflow when a new form submission arrives
 */

// Configuration - UPDATE THESE VALUES
const CONFIG = {
  GITHUB_TOKEN: 'YOUR_GITHUB_PAT_SECRET', // Will be stored securely
  REPO_OWNER: 'samuelmatia',
  REPO_NAME: 'shift-gnn-challenge',
  WORKFLOW_FILE: 'process_google_form_polling.yml'
};

/**
 * Main function triggered when form is submitted
 */
function onFormSubmit(e) {
  Logger.log('Form submission detected, triggering workflow...');
  
  // Get GitHub token from Properties (secure storage)
  const scriptProperties = PropertiesService.getScriptProperties();
  const githubToken = scriptProperties.getProperty('GITHUB_PAT');
  
  if (!githubToken) {
    Logger.log('ERROR: GITHUB_PAT not set in script properties');
    sendErrorEmail('GitHub PAT not configured');
    return;
  }
  
  // Trigger GitHub Actions workflow
  const success = triggerWorkflow(githubToken);
  
  if (success) {
    Logger.log('✅ Workflow triggered successfully');
  } else {
    Logger.log('❌ Failed to trigger workflow');
    sendErrorEmail('Failed to trigger GitHub workflow');
  }
}

/**
 * Trigger GitHub Actions workflow via API
 */
function triggerWorkflow(githubToken) {
  const url = `https://api.github.com/repos/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/actions/workflows/${CONFIG.WORKFLOW_FILE}/dispatches`;
  
  const payload = {
    'ref': 'main'
  };
  
  const options = {
    'method': 'post',
    'headers': {
      'Authorization': `token ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'Google-Apps-Script'
    },
    'payload': JSON.stringify(payload),
    'muteHttpExceptions': true
  };
  
  try {
    const response = UrlFetchApp.fetch(url, options);
    const statusCode = response.getResponseCode();
    const responseText = response.getContentText();
    
    Logger.log(`Response status: ${statusCode}`);
    Logger.log(`Response: ${responseText}`);
    
    if (statusCode === 204) {
      // 204 No Content = success
      return true;
    } else {
      Logger.log(`Unexpected status code: ${statusCode}`);
      return false;
    }
  } catch (error) {
    Logger.log(`Error triggering workflow: ${error.toString()}`);
    return false;
  }
}

/**
 * Send error notification email (optional)
 */
function sendErrorEmail(message) {
  // Uncomment and configure if you want email notifications
  /*
  const email = 'your-email@example.com';
  const subject = 'Shift-GNN Challenge: Workflow Trigger Error';
  const body = `Error: ${message}\n\nTime: ${new Date().toISOString()}`;
  MailApp.sendEmail(email, subject, body);
  */
}

/**
 * Setup function - Run this ONCE to configure the GitHub PAT
 * This stores the token securely in Script Properties
 */
function setupGitHubPAT() {
  // Replace with your actual GitHub Personal Access Token
  const githubPAT = 'YOUR_GITHUB_PAT_HERE';
  
  if (githubPAT === 'YOUR_GITHUB_PAT_HERE') {
    Logger.log('ERROR: Please replace YOUR_GITHUB_PAT_HERE with your actual token');
    return;
  }
  
  const scriptProperties = PropertiesService.getScriptProperties();
  scriptProperties.setProperty('GITHUB_PAT', githubPAT);
  Logger.log('✅ GitHub PAT stored securely');
  Logger.log('You can now delete this function or comment it out');
}

/**
 * Test function - Manually trigger workflow (for testing)
 */
function testWorkflowTrigger() {
  Logger.log('Testing workflow trigger...');
  onFormSubmit(null);
}
```

5. **Update the configuration**:
   - Replace `YOUR_GITHUB_PAT_HERE` in the `setupGitHubPAT()` function with your actual GitHub PAT
   - Verify `REPO_OWNER` and `REPO_NAME` are correct

6. **Save the script**: Click **File** → **Save** (or `Ctrl+S` / `Cmd+S`)
   - Name it: `Trigger GitHub Workflow`

---

## Step 4: Store GitHub PAT Securely

1. In the Apps Script editor, click **Run** → Select `setupGitHubPAT`
2. Click **Run** (the play button)
3. **Authorize** the script when prompted:
   - Click **"Review permissions"**
   - Choose your Google account
   - Click **"Advanced"** → **"Go to [Project Name] (unsafe)"**
   - Click **"Allow"**
4. Check the **Execution log** (View → Logs):
   - You should see: `✅ GitHub PAT stored securely`
5. **⚠️ IMPORTANT**: After setup, delete or comment out the `setupGitHubPAT()` function for security

---

## Step 5: Set Up Form Submit Trigger

1. In Apps Script editor, click **Triggers** (clock icon on the left)
2. Click **"+ Add Trigger"** (bottom right)
3. Configure:
   - **Choose which function to run**: `onFormSubmit`
   - **Select event source**: `From form`
   - **Select event type**: `On form submit`
   - **Failure notification settings**: Choose your preference
4. Click **"Save"**
5. **Authorize** if prompted again

---

## Step 6: Test the Setup

### Option A: Test via Apps Script
1. In Apps Script editor, click **Run** → Select `testWorkflowTrigger`
2. Click **Run**
3. Check the execution log for success message
4. Go to GitHub → **Actions** → Check if workflow was triggered

### Option B: Test via Form Submission
1. Submit a test entry via your Google Form
2. Wait 10-30 seconds
3. Go to GitHub → **Actions** → Check if workflow was triggered automatically

---

## Step 7: Verify Workflow Triggering

1. Go to GitHub → **Actions** → **"Process Google Form Submissions (Polling)"**
2. You should see a new workflow run triggered by `repository_dispatch`
3. The workflow should process the submission and update the leaderboard

---

## Troubleshooting

### Workflow not triggering
- ✅ Check that `GITHUB_PAT` is stored in Script Properties (run `setupGitHubPAT` again)
- ✅ Verify trigger is set up correctly (Triggers → Check `onFormSubmit` trigger exists)
- ✅ Check execution log in Apps Script for errors
- ✅ Verify GitHub PAT has `repo` and `workflow` scopes
- ✅ Check GitHub Actions tab for any failed workflow runs

### "401 Unauthorized" error
- GitHub PAT is incorrect or expired
- Regenerate PAT and run `setupGitHubPAT()` again

### "404 Not Found" error
- Check `REPO_OWNER`, `REPO_NAME`, and `WORKFLOW_FILE` are correct
- Verify workflow file exists: `.github/workflows/process_google_form_polling.yml`

### Trigger not firing on form submit
- Ensure trigger is set to `onFormSubmit` with event type `On form submit`
- Check that the Google Sheet is linked to the Google Form
- Try submitting a test form entry

---

## Security Best Practices

1. **Never commit GitHub PAT to code**: Always use Script Properties
2. **Delete `setupGitHubPAT()` function** after setup (or comment it out)
3. **Use token with minimal scopes**: Only `repo` and `workflow`
4. **Set token expiration**: Don't use "no expiration" unless necessary
5. **Rotate tokens periodically**: Regenerate PAT every 90 days

---

## Optional: Keep Scheduled Workflow as Backup

You can keep the scheduled workflow (every 15 minutes) as a backup in case the webhook fails:

- The scheduled workflow will still run periodically
- The webhook triggers instantly when form is submitted
- Both can coexist without conflicts

---

## Result

✅ **Instant workflow triggering** when form is submitted  
✅ **No waiting** for scheduled intervals  
✅ **More reliable** than scheduled workflows  
✅ **Only runs when needed** (when form is submitted)

---

## Next Steps

1. Complete all setup steps above
2. Test with a form submission
3. Verify workflow runs in GitHub Actions
4. Monitor for a few days to ensure reliability
5. (Optional) Remove or increase scheduled workflow interval if webhook works well
