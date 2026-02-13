/**
 * Google Apps Script: Trigger GitHub Actions Workflow on Form Submit
 * 
 * Instructions:
 * 1. Open your Google Sheet (linked to Google Form)
 * 2. Extensions → Apps Script
 * 3. Paste this code
 * 4. Update CONFIG section with your values
 * 5. Run setupGitHubPAT() once to store your token
 * 6. Set up trigger: Triggers → Add Trigger → onFormSubmit → On form submit
 */

// ============================================================================
// CONFIGURATION - UPDATE THESE VALUES
// ============================================================================

const CONFIG = {
  // Your GitHub repository details
  REPO_OWNER: 'samuelmatia',
  REPO_NAME: 'shift-gnn-challenge',
  WORKFLOW_FILE: 'process_google_form_polling.yml', // Name of workflow file
  
  // GitHub Personal Access Token (will be stored securely)
  // Get token from: GitHub → Settings → Developer settings → Personal access tokens
  // Required scopes: repo, workflow
  GITHUB_PAT: '' // Leave empty - will be set via setupGitHubPAT()
};

// ============================================================================
// MAIN FUNCTION - Triggered automatically when form is submitted
// ============================================================================

/**
 * Triggered when a new form submission arrives
 */
function onFormSubmit(e) {
  Logger.log('📝 Form submission detected');
  Logger.log('⏱️  Time: ' + new Date().toISOString());
  
  // Get GitHub token from secure storage
  const scriptProperties = PropertiesService.getScriptProperties();
  const githubToken = scriptProperties.getProperty('GITHUB_PAT');
  
  if (!githubToken) {
    const errorMsg = '❌ ERROR: GitHub PAT not configured. Run setupGitHubPAT() first.';
    Logger.log(errorMsg);
    return;
  }
  
  // Trigger GitHub Actions workflow
  Logger.log('🚀 Triggering GitHub Actions workflow...');
  const success = triggerGitHubWorkflow(githubToken);
  
  if (success) {
    Logger.log('✅ Workflow triggered successfully!');
    Logger.log('📊 Check GitHub Actions: https://github.com/' + CONFIG.REPO_OWNER + '/' + CONFIG.REPO_NAME + '/actions');
  } else {
    Logger.log('❌ Failed to trigger workflow');
  }
}

// ============================================================================
// GITHUB API FUNCTIONS
// ============================================================================

/**
 * Trigger GitHub Actions workflow via GitHub API
 * 
 * Note: Requires Personal Access Token (classic) with 'repo' and 'workflow' scopes
 * OR use repository_dispatch event type which requires only 'repo' scope
 */
function triggerGitHubWorkflow(githubToken) {
  // Method 1: Try workflow_dispatch (requires 'repo' and 'workflow' scopes)
  // Method 2: Fallback to repository_dispatch (requires only 'repo' scope)
  
  // Try workflow_dispatch first
  const workflowUrl = `https://api.github.com/repos/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/actions/workflows/${CONFIG.WORKFLOW_FILE}/dispatches`;
  
  const workflowPayload = {
    'ref': 'main'
  };
  
  const workflowOptions = {
    'method': 'post',
    'headers': {
      'Authorization': `token ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'Google-Apps-Script-Webhook'
    },
    'payload': JSON.stringify(workflowPayload),
    'muteHttpExceptions': true
  };
  
  try {
    const response = UrlFetchApp.fetch(workflowUrl, workflowOptions);
    const statusCode = response.getResponseCode();
    const responseText = response.getContentText();
    
    Logger.log(`📡 HTTP Status: ${statusCode}`);
    
    if (statusCode === 204 || statusCode === 200 || statusCode === 201) {
      Logger.log('✅ Workflow triggered successfully via workflow_dispatch');
      return true;
    } else if (statusCode === 403) {
      // 403 = Permission denied, try repository_dispatch instead
      Logger.log('⚠️  workflow_dispatch failed (403). Trying repository_dispatch...');
      return triggerRepositoryDispatch(githubToken);
    } else {
      // Other error
      Logger.log(`❌ Error response: ${responseText}`);
      try {
        const errorJson = JSON.parse(responseText);
        if (errorJson.message) {
          Logger.log(`Error message: ${errorJson.message}`);
          // If it's a permission error, try repository_dispatch
          if (errorJson.message.includes('not accessible') || statusCode === 403) {
            Logger.log('⚠️  Trying repository_dispatch as fallback...');
            return triggerRepositoryDispatch(githubToken);
          }
        }
      } catch (e) {
        // Not JSON, ignore
      }
      return false;
    }
  } catch (error) {
    Logger.log(`❌ Exception: ${error.toString()}`);
    Logger.log(`⚠️  Trying repository_dispatch as fallback...`);
    return triggerRepositoryDispatch(githubToken);
  }
}

/**
 * Alternative method: Use repository_dispatch event (requires only 'repo' scope)
 */
function triggerRepositoryDispatch(githubToken) {
  const url = `https://api.github.com/repos/${CONFIG.REPO_OWNER}/${CONFIG.REPO_NAME}/dispatches`;
  
  const payload = {
    'event_type': 'process-submissions',
    'client_payload': {
      'source': 'google_form',
      'timestamp': new Date().toISOString()
    }
  };
  
  const options = {
    'method': 'post',
    'headers': {
      'Authorization': `token ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'Google-Apps-Script-Webhook'
    },
    'payload': JSON.stringify(payload),
    'muteHttpExceptions': true
  };
  
  try {
    const response = UrlFetchApp.fetch(url, options);
    const statusCode = response.getResponseCode();
    const responseText = response.getContentText();
    
    Logger.log(`📡 repository_dispatch HTTP Status: ${statusCode}`);
    
    if (statusCode === 204 || statusCode === 200 || statusCode === 201) {
      Logger.log('✅ Workflow triggered successfully via repository_dispatch');
      return true;
    } else {
      Logger.log(`❌ repository_dispatch error: ${responseText}`);
      try {
        const errorJson = JSON.parse(responseText);
        if (errorJson.message) {
          Logger.log(`Error message: ${errorJson.message}`);
        }
      } catch (e) {
        // Not JSON
      }
      return false;
    }
  } catch (error) {
    Logger.log(`❌ repository_dispatch exception: ${error.toString()}`);
    return false;
  }
}

// ============================================================================
// SETUP FUNCTION - Run this ONCE to configure GitHub PAT
// ============================================================================

/**
 * Setup function - Run this ONCE to store your GitHub Personal Access Token
 * 
 * Steps:
 * 1. Replace YOUR_GITHUB_PAT_HERE below with your actual token
 * 2. Click Run → Select setupGitHubPAT → Click Run
 * 3. Authorize the script when prompted
 * 4. Check logs for success message
 * 5. DELETE or COMMENT OUT this function after setup (for security)
 */
function setupGitHubPAT() {
  // ⚠️ REPLACE THIS WITH YOUR ACTUAL GITHUB PERSONAL ACCESS TOKEN
  const githubPAT = 'YOUR_GITHUB_PAT_HERE';
  
  // Validation
  if (!githubPAT || githubPAT === 'YOUR_GITHUB_PAT_HERE') {
    Logger.log('❌ ERROR: Please replace YOUR_GITHUB_PAT_HERE with your actual GitHub PAT');
    Logger.log('📖 Get token from: GitHub → Settings → Developer settings → Personal access tokens');
    return;
  }
  
  if (!githubPAT.startsWith('ghp_')) {
    Logger.log('⚠️  WARNING: GitHub PAT should start with "ghp_". Please verify your token.');
  }
  
  // Store token securely in Script Properties
  const scriptProperties = PropertiesService.getScriptProperties();
  scriptProperties.setProperty('GITHUB_PAT', githubPAT);
  
  Logger.log('✅ GitHub PAT stored securely in Script Properties');
  Logger.log('🔒 Token is now encrypted and stored safely');
  Logger.log('⚠️  IMPORTANT: Delete or comment out this function after setup');
  
  // Test the token by trying to trigger workflow
  Logger.log('🧪 Testing token...');
  const testSuccess = triggerGitHubWorkflow(githubPAT);
  if (testSuccess) {
    Logger.log('✅ Token test successful! Setup complete.');
  } else {
    Logger.log('❌ Token test failed. Please verify your token has correct scopes (repo, workflow).');
  }
}

// ============================================================================
// TEST FUNCTION - For manual testing
// ============================================================================

/**
 * Test function - Manually trigger workflow (for testing without form submission)
 */
function testWorkflowTrigger() {
  Logger.log('🧪 Testing workflow trigger manually...');
  Logger.log('⏱️  Time: ' + new Date().toISOString());
  onFormSubmit(null);
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Check if GitHub PAT is configured
 */
function checkConfiguration() {
  const scriptProperties = PropertiesService.getScriptProperties();
  const githubToken = scriptProperties.getProperty('GITHUB_PAT');
  
  if (githubToken) {
    Logger.log('✅ GitHub PAT is configured');
    Logger.log('🔒 Token length: ' + githubToken.length + ' characters');
    Logger.log('🔒 Token starts with: ' + githubToken.substring(0, 4) + '...');
  } else {
    Logger.log('❌ GitHub PAT is NOT configured');
    Logger.log('📖 Run setupGitHubPAT() to configure');
  }
  
  Logger.log('📋 Repository: ' + CONFIG.REPO_OWNER + '/' + CONFIG.REPO_NAME);
  Logger.log('📋 Workflow: ' + CONFIG.WORKFLOW_FILE);
}

/**
 * Clear stored GitHub PAT (for security/maintenance)
 */
function clearGitHubPAT() {
  const scriptProperties = PropertiesService.getScriptProperties();
  scriptProperties.deleteProperty('GITHUB_PAT');
  Logger.log('✅ GitHub PAT cleared from storage');
}
