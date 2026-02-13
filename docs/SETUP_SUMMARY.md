# Setup Summary: Google Form Submission System with Automated Leaderboard Updates

## Overview
This document summarizes the steps taken to set up an automated submission and leaderboard update system using Google Forms, Google Sheets, Google Drive, and GitHub Actions.

---

## Step 1: Google Form Creation
- Created a Google Form with required fields:
  - Team Name (required)
  - Model Type (required: human, llm, or human+llm)
  - Submission File (.csv) (required: file upload)
- Configured the form to collect responses in a Google Sheet

---

## Step 2: Google Cloud API Setup
- Created a Google Cloud Project
- Enabled Google Sheets API and Google Drive API
- Created a Service Account
- Generated a JSON key file (`shift-gnn-challenge-16efdb090a61.json`)
- Shared the Google Sheet with the Service Account email (Editor access)
- Shared the Google Drive folder containing CSV submissions with the Service Account email (Editor access)

---

## Step 3: GitHub Secrets Configuration
Configured the following secrets in GitHub (Settings → Secrets and variables → Actions):

- **`GOOGLE_SHEETS_ID`**: Google Sheet ID (extracted from Sheet URL)
- **`GOOGLE_CREDENTIALS_JSON`**: Complete content of the Service Account JSON file
- **`GOOGLE_DRIVE_FILE_ID`**: File ID of the private test data (`test.parquet`) on Google Drive
- **`PRIVATE_DATA_METHOD`**: Set to `google_drive` (default)

---

## Step 4: Script Development
Created and configured the following scripts:

- **`scripts/process_google_form_submissions.py`**: 
  - Reads submissions from Google Sheets
  - Downloads CSV files from Google Drive
  - Evaluates submissions using `scoring_script.py`
  - Updates `leaderboard.json` and `leaderboard.html`
  - Pushes changes to GitHub repository

- **`scripts/download_private_data.py`**: 
  - Downloads private test data from Google Drive using Service Account credentials
  - Supports multiple methods (Google Drive, URL, S3)

---

## Step 5: GitHub Actions Workflow Configuration
Created `.github/workflows/process_google_form_polling.yml`:

- **Trigger**: Scheduled every 5 minutes (`*/5 * * * *`) + manual dispatch
- **Permissions**: `contents: write` (allows pushing to repository)
- **Steps**:
  1. Checkout repository
  2. Setup Python 3.9
  3. Install dependencies (Google API clients, pandas, scikit-learn, requests)
  4. Download private test data from Google Drive
  5. Process Google Form submissions and push leaderboard updates

---

## Step 6: File Management
- Updated `.gitignore` to exclude sensitive files while allowing necessary scripts:
  - Excluded: `data/private/`, `data/raw_data/`, credentials JSON files
  - Allowed: Scripts needed for CI/CD (`process_google_form_submissions.py`, `evaluate_all_submissions.py`, etc.)

---

## Step 7: Bug Fixes and Improvements
Resolved several issues during setup:

1. **Missing script in repository**: Added `process_google_form_submissions.py` to Git
2. **Filename sanitization**: Fixed timestamp handling to replace slashes (`/`) with dashes (`-`) in filenames
3. **Private data download**: Enhanced `download_private_data.py` to use Service Account credentials
4. **Git push permissions**: Added `permissions: contents: write` to workflow
5. **Error handling**: Improved error messages and logging throughout the pipeline

---

## Step 8: Testing and Verification
- Tested manual workflow execution via GitHub Actions UI
- Verified automatic execution every 5 minutes
- Confirmed leaderboard updates are pushed to the repository
- Validated that submissions are processed correctly

---

## Final Architecture

```
Google Form → Google Sheets → GitHub Actions (every 5 min)
                                      ↓
                            Process Submissions
                                      ↓
                            Download CSV from Drive
                                      ↓
                            Evaluate (using test.parquet)
                                      ↓
                            Update leaderboard.json/html
                                      ↓
                            Push to GitHub Repository
                                      ↓
                            GitHub Pages (public leaderboard)
```

---

## Key Files Created/Modified

- `.github/workflows/process_google_form_polling.yml` - Automated workflow
- `scripts/process_google_form_submissions.py` - Main processing script
- `scripts/download_private_data.py` - Private data downloader
- `.gitignore` - Updated to exclude sensitive files
- `docs/GOOGLE_FORM_SETUP.md` - Detailed setup guide
- `docs/AUTOMATION_SETUP.md` - Automation configuration guide
- `docs/PROCHAINES_ETAPES.md` - Next steps checklist (French)

---

## Result
✅ Automated submission processing system fully operational
✅ Leaderboard updates automatically every 5 minutes
✅ Private submissions remain confidential (only scores appear on public leaderboard)
✅ One submission per participant enforced
✅ No manual intervention required
