"""
Process submissions from Google Form and update leaderboard.
This script should run in a private environment (private repo CI or local server).

Workflow:
1. Read new submissions from Google Sheets (linked to Google Form)
2. Download CSV files from Google Drive
3. Evaluate each submission
4. Update leaderboard.json/html with scores and ranks only
5. Push only leaderboard files to public repo

Requirements:
- google-api-python-client
- google-auth-httplib2
- google-auth-oauthlib
- gitpython (or use subprocess for git)
"""

import os
import json
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    import io
    GOOGLE_API_AVAILABLE = True
except ImportError:
    GOOGLE_API_AVAILABLE = False
    print("Warning: Google API libraries not installed. Install with:")
    print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")

from scripts.evaluate_all_submissions import evaluate_submission
from scripts.generate_leaderboard import assign_kaggle_ranks, generate_html


def load_google_credentials(credentials_path=None):
    """Load Google service account credentials."""
    if not GOOGLE_API_AVAILABLE:
        raise ImportError("Google API libraries not installed")
    
    # Priority: credentials_path > GOOGLE_CREDENTIALS_PATH > GOOGLE_CREDENTIALS_JSON
    creds_path = credentials_path or os.getenv('GOOGLE_CREDENTIALS_PATH')
    creds_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
    
    # If JSON string provided, create temp file
    if creds_json and not creds_path:
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            f.write(creds_json)
            creds_path = f.name
    
    if not creds_path or not Path(creds_path).exists():
        raise FileNotFoundError(
            "Google credentials not found. Set GOOGLE_CREDENTIALS_PATH or GOOGLE_CREDENTIALS_JSON"
        )
    
    credentials = service_account.Credentials.from_service_account_file(
        creds_path,
        scopes=['https://www.googleapis.com/auth/spreadsheets.readonly',
                'https://www.googleapis.com/auth/drive.readonly']
    )
    return credentials


def get_sheet_names(sheets_id, credentials):
    """Get list of sheet names in the spreadsheet."""
    service = build('sheets', 'v4', credentials=credentials)
    spreadsheet = service.spreadsheets().get(spreadsheetId=sheets_id).execute()
    return [sheet['properties']['title'] for sheet in spreadsheet.get('sheets', [])]


def get_google_form_responses(sheets_id, credentials, sheet_name=None):
    """
    Read responses from Google Sheets (linked to Google Form).
    
    Args:
        sheets_id: Google Sheets ID
        credentials: Google credentials
        sheet_name: Name of the sheet (if None, uses first sheet)
    
    Returns:
        List of dicts with form responses
    """
    service = build('sheets', 'v4', credentials=credentials)
    sheet = service.spreadsheets()
    
    # If sheet_name not provided, get the first sheet
    if sheet_name is None:
        sheet_names = get_sheet_names(sheets_id, credentials)
        if not sheet_names:
            raise ValueError("No sheets found in the spreadsheet")
        sheet_name = sheet_names[0]
        print(f"Using sheet: '{sheet_name}'")
    
    # Read all rows
    try:
        result = sheet.values().get(
            spreadsheetId=sheets_id,
            range=f'{sheet_name}!A:Z'
        ).execute()
    except Exception as e:
        # Try to list available sheets for debugging
        available_sheets = get_sheet_names(sheets_id, credentials)
        print(f"Error reading sheet '{sheet_name}'. Available sheets: {available_sheets}")
        raise
    
    values = result.get('values', [])
    if not values:
        return []
    
    # First row is headers
    headers = values[0]
    responses = []
    
    for row in values[1:]:
        if not row:  # Skip empty rows
            continue
        response = {}
        for i, header in enumerate(headers):
            response[header] = row[i] if i < len(row) else ''
        responses.append(response)
    
    return responses


def download_file_from_drive(file_id, credentials, output_path):
    """Download a file from Google Drive by file ID."""
    service = build('drive', 'v3', credentials=credentials)
    
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    
    done = False
    while done is False:
        status, done = downloader.next_chunk()
    
    fh.seek(0)
    with open(output_path, 'wb') as f:
        f.write(fh.read())


def extract_file_id_from_drive_link(link):
    """
    Extract file ID from Google Drive share link.
    Formats:
    - https://drive.google.com/file/d/FILE_ID/view
    - https://drive.google.com/open?id=FILE_ID
    - https://drive.google.com/file/d/FILE_ID/edit
    - Just the file ID itself (if already extracted)
    """
    if not link:
        return None
    
    # If it's already just an ID (no URL), return as-is
    if len(link) < 50 and '/' not in link and '?' not in link:
        return link
    
    if '/file/d/' in link:
        return link.split('/file/d/')[1].split('/')[0]
    elif 'id=' in link:
        return link.split('id=')[1].split('&')[0].split('#')[0]
    elif '/open?id=' in link:
        return link.split('/open?id=')[1].split('&')[0].split('#')[0]
    
    return None


def process_submissions_from_google_form(
    sheets_id,
    credentials_path=None,
    submissions_dir='submissions',
    processed_file='.processed_submissions.json',
    sheet_name=None
):
    """
    Main function to process Google Form submissions.
    
    Args:
        sheets_id: Google Sheets ID (from Sheets URL)
        credentials_path: Path to Google service account JSON
        submissions_dir: Directory to save downloaded CSV files
        processed_file: File to track already processed submissions
    """
    print("="*60)
    print("Processing Google Form Submissions")
    print("="*60)
    
    # Load credentials
    credentials = load_google_credentials(credentials_path)
    
    # Load list of already processed submissions
    processed_path = Path(__file__).parent.parent / processed_file
    processed = set()
    if processed_path.exists():
        with open(processed_path, 'r') as f:
            data = json.load(f)
            processed = set(data.get('processed_timestamps', []))
    
    # Get form responses
    print(f"\nReading responses from Google Sheets: {sheets_id}")
    responses = get_google_form_responses(sheets_id, credentials, sheet_name=sheet_name)
    print(f"Found {len(responses)} total responses")
    
    # Debug: Show available columns (from headers even if no responses)
    if responses:
        print(f"\nAvailable columns in form responses:")
        for key in responses[0].keys():
            print(f"  - '{key}'")
    else:
        # Still show headers if available (read directly from sheet)
        print("\nNo responses found. Checking sheet structure...")
        try:
            service = build('sheets', 'v4', credentials=credentials)
            sheet = service.spreadsheets()
            result = sheet.values().get(
                spreadsheetId=sheets_id,
                range=f'{sheet_name}!A1:Z1'  # Just headers
            ).execute()
            headers = result.get('values', [[]])[0]
            if headers:
                print("Column headers found:")
                for i, header in enumerate(headers):
                    print(f"  Column {i+1}: '{header}'")
        except Exception as e:
            print(f"Could not read headers: {e}")
    
    # Filter new responses (by timestamp)
    # Column name: "Horodateur" (French) or "Timestamp" (English)
    new_responses = []
    for resp in responses:
        timestamp = resp.get('Horodateur', resp.get('Timestamp', resp.get('timestamp', '')))
        if timestamp and timestamp not in processed:
            new_responses.append(resp)
    
    if not new_responses:
        print("No new submissions to process")
        return
    
    print(f"Processing {len(new_responses)} new submission(s)")
    
    # Create submissions directory
    submissions_path = Path(__file__).parent.parent / submissions_dir
    submissions_path.mkdir(exist_ok=True)
    
    # Process each submission
    evaluation_results = []
    new_processed = []
    
    for resp in new_responses:
        # Exact column names from your Google Sheet (French)
        timestamp = resp.get('Horodateur', resp.get('Timestamp', resp.get('timestamp', '')))
        email = resp.get('Adresse e-mail', resp.get('Email Address', resp.get('Email', resp.get('email', ''))))
        team_name = resp.get('1. Team Name', resp.get('Team Name', resp.get('team_name', '')))
        model_type = resp.get('2. Model Type', resp.get('Model Type', resp.get('model_type', 'unknown'))).lower()
        csv_link = resp.get('3. Submission File ( .csv)', resp.get('Submission File ( .csv)', resp.get('CSV File', '')))
        
        if not team_name:
            print(f"Warning: Skipping response without team name (timestamp: {timestamp})")
            continue
        
        if not csv_link:
            print(f"Warning: Skipping {team_name} - no CSV file link")
            continue
        
        print(f"\nProcessing submission from: {team_name} ({email})")
        
        # Extract file ID from Google Drive link
        # Google Forms stores file uploads as Drive links or file IDs
        file_id = extract_file_id_from_drive_link(csv_link)
        if not file_id:
            print(f"  Error: Could not extract file ID from link: {csv_link}")
            print(f"  Debug: csv_link value = '{csv_link}'")
            print(f"  Hint: Google Forms stores file uploads as Google Drive links")
            continue
        
        # Download CSV file
        csv_filename = f"{team_name}_{timestamp.replace(' ', '_').replace(':', '-')}.csv"
        csv_path = submissions_path / csv_filename
        
        try:
            print(f"  Downloading CSV from Google Drive (ID: {file_id})...")
            download_file_from_drive(file_id, credentials, csv_path)
            print(f"  Downloaded to: {csv_path}")
        except Exception as e:
            print(f"  Error downloading file: {e}")
            continue
        
        # Evaluate submission
        print(f"  Evaluating submission...")
        scores = evaluate_submission(str(csv_path))
        
        if scores:
            result = {
                'team': team_name,
                'file': str(csv_path),
                'scores': scores,
                'model_type': model_type,
                'email': email,
                'timestamp': timestamp
            }
            evaluation_results.append(result)
            new_processed.append(timestamp)
            print(f"  ✓ Evaluation complete: Weighted F1 = {scores.get('weighted_f1', 0):.4f}")
        else:
            print(f"  ✗ Evaluation failed")
    
    # Update leaderboard if we have new results
    if evaluation_results:
        print(f"\n{'='*60}")
        print("Updating leaderboard...")
        print(f"{'='*60}")
        
        # Save evaluation results
        results_file = Path(__file__).parent.parent / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2)
        
        # Generate leaderboard
        from scripts.generate_leaderboard import generate_leaderboard
        leaderboard = generate_leaderboard()
        
        print(f"\n✓ Leaderboard updated with {len(leaderboard.get('submissions', []))} teams")
        
        # Save processed timestamps
        all_processed = list(processed) + new_processed
        with open(processed_path, 'w') as f:
            json.dump({'processed_timestamps': all_processed}, f, indent=2)
        
        print(f"\n✓ Processed {len(new_processed)} new submission(s)")
        print("\nNext step: Push leaderboard.json and leaderboard.html to public repo")
    else:
        print("\nNo valid submissions to add to leaderboard")


def push_leaderboard_to_repo(repo_path=None, remote='origin', branch='main'):
    """
    Push only leaderboard.json and leaderboard.html to the public repo.
    This should be called after process_submissions_from_google_form.
    """
    repo_path = repo_path or Path(__file__).parent.parent
    
    print(f"\n{'='*60}")
    print("Pushing leaderboard to public repo")
    print(f"{'='*60}")
    
    # Check if leaderboard files exist
    leaderboard_json = repo_path / 'leaderboard.json'
    leaderboard_html = repo_path / 'leaderboard.html'
    
    if not leaderboard_json.exists() or not leaderboard_html.exists():
        print("Error: leaderboard.json or leaderboard.html not found")
        return False
    
    try:
        # Add only leaderboard files
        subprocess.run(['git', 'add', 'leaderboard.json', 'leaderboard.html'], 
                      cwd=repo_path, check=True)
        
        # Check if there are changes
        result = subprocess.run(['git', 'diff', '--staged', '--quiet'], 
                               cwd=repo_path)
        if result.returncode == 0:
            print("No changes to leaderboard files")
            return True
        
        # Commit
        subprocess.run(['git', 'commit', '-m', 
                       f'Update leaderboard from Google Form submissions [{datetime.now().strftime("%Y-%m-%d %H:%M")}]'],
                      cwd=repo_path, check=True)
        
        # Push
        subprocess.run(['git', 'push', remote, branch], 
                      cwd=repo_path, check=True)
        
        print("✓ Leaderboard pushed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error pushing to repo: {e}")
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Process Google Form submissions')
    parser.add_argument('--sheets-id', required=True,
                       help='Google Sheets ID (from Sheets URL)')
    parser.add_argument('--credentials', 
                       help='Path to Google service account JSON file')
    parser.add_argument('--sheet-name',
                       help='Name of the sheet to read (default: first sheet)')
    parser.add_argument('--push', action='store_true',
                       help='Push leaderboard to repo after processing')
    parser.add_argument('--remote', default='origin',
                       help='Git remote name (default: origin)')
    parser.add_argument('--branch', default='main',
                       help='Git branch name (default: main)')
    
    args = parser.parse_args()
    
    # Process submissions
    process_submissions_from_google_form(
        sheets_id=args.sheets_id,
        credentials_path=args.credentials,
        sheet_name=args.sheet_name
    )
    
    # Push if requested
    if args.push:
        push_leaderboard_to_repo(remote=args.remote, branch=args.branch)
