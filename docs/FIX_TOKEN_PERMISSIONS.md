# Fix: GitHub Token 403 Error

## Problem
Error 403: "Resource not accessible by personal access token"

This means your GitHub Personal Access Token doesn't have the required permissions.

---

## Solution: Create Token with Correct Scopes

### Option 1: Use Classic Token with 'repo' and 'workflow' Scopes (Recommended)

1. **Go to GitHub Token Settings**:
   - https://github.com/settings/tokens
   - Click **"Generate new token"** → **"Generate new token (classic)"**

2. **Configure Token**:
   - **Note**: `Shift-GNN Webhook`
   - **Expiration**: Your choice
   - **Scopes**: Check these EXACT boxes:
     - ✅ **`repo`** (Full control of private repositories)
       - This includes: `repo:status`, `repo_deployment`, `public_repo`, `repo:invite`, `security_events`
     - ✅ **`workflow`** (Update GitHub Action workflows)
   
   **IMPORTANT**: Both `repo` AND `workflow` must be checked!

3. **Generate and Copy Token**

4. **Update Apps Script**:
   - Replace token in `setupGitHubPAT()` function
   - Run `setupGitHubPAT()` again

---

### Option 2: Use Repository Dispatch (Easier - Only 'repo' Scope Needed)

The script now automatically falls back to `repository_dispatch` if `workflow_dispatch` fails. This only requires `repo` scope.

**Steps**:
1. Create token with **only** `repo` scope (no need for `workflow`)
2. The script will automatically use `repository_dispatch` method
3. This should work with just `repo` scope

---

## Verify Token Scopes

After creating the token, you can verify it has the right scopes:

1. Go to: https://github.com/settings/tokens
2. Find your token
3. Check that it shows:
   - ✅ `repo` scope
   - ✅ `workflow` scope (if using workflow_dispatch)

---

## Common Issues

### "Token should start with ghp_"
- Classic tokens start with `ghp_`
- Fine-grained tokens start with `github_pat_`
- **Use classic token** for this use case

### "403 Forbidden"
- Token missing `repo` scope → Add `repo` scope
- Token missing `workflow` scope → Add `workflow` scope OR use repository_dispatch
- Token expired → Generate new token
- Repository restrictions → Check repository settings

### Repository Has Token Restrictions
If your repository has token restrictions:
1. Go to repository → **Settings** → **Actions** → **General**
2. Scroll to **"Workflow permissions"**
3. Ensure **"Read and write permissions"** is selected
4. Check **"Allow GitHub Actions to create and approve pull requests"**

---

## Updated Script Behavior

The script now:
1. **First tries** `workflow_dispatch` (requires `repo` + `workflow`)
2. **Falls back** to `repository_dispatch` if 403 error (requires only `repo`)
3. **Logs** which method was used

---

## Quick Fix Steps

1. **Delete old token** (if you want to start fresh)
2. **Create new classic token** with:
   - ✅ `repo` scope (required)
   - ✅ `workflow` scope (for workflow_dispatch, optional if using repository_dispatch)
3. **Update Apps Script**:
   - Replace token in `setupGitHubPAT()`
   - Run `setupGitHubPAT()` again
4. **Test**: Run `testWorkflowTrigger()` function

---

## Test Token Permissions

You can test if your token has the right permissions:

```javascript
function testTokenPermissions() {
  const scriptProperties = PropertiesService.getScriptProperties();
  const githubToken = scriptProperties.getProperty('GITHUB_PAT');
  
  if (!githubToken) {
    Logger.log('❌ Token not configured');
    return;
  }
  
  // Test repo access
  const repoUrl = `https://api.github.com/repos/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}`;
  const options = {
    'headers': {
      'Authorization': `token ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json'
    },
    'muteHttpExceptions': true
  };
  
  const response = UrlFetchApp.fetch(repoUrl, options);
  Logger.log(`Repo access test: ${response.getResponseCode()}`);
  
  if (response.getResponseCode() === 200) {
    Logger.log('✅ Token has repo access');
  } else {
    Logger.log('❌ Token missing repo scope');
  }
}
```

Run this function to verify your token has `repo` scope at minimum.
