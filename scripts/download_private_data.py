"""
Script to download private test data from a secure location.
This script is used by GitHub Actions to access test labels.
"""
import os
import sys
import json
from pathlib import Path

def download_from_google_drive(file_id, output_path, access_token=None, credentials_json=None):
    """Download file from Google Drive using file ID."""
    try:
        # Method 1: Using Google API with Service Account credentials
        if credentials_json:
            try:
                from google.oauth2 import service_account
                from googleapiclient.discovery import build
                from googleapiclient.http import MediaIoBaseDownload
                import io
                
                # Parse credentials JSON
                if isinstance(credentials_json, str):
                    creds_dict = json.loads(credentials_json)
                else:
                    creds_dict = credentials_json
                
                # Create credentials from service account
                credentials = service_account.Credentials.from_service_account_info(
                    creds_dict,
                    scopes=['https://www.googleapis.com/auth/drive.readonly']
                )
                
                # Build Drive API service
                service = build('drive', 'v3', credentials=credentials)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Download file
                request = service.files().get_media(fileId=file_id)
                fh = io.FileIO(output_path, 'wb')
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while done is False:
                    status, done = downloader.next_chunk()
                    if status:
                        print(f"Download progress: {int(status.progress() * 100)}%")
                
                print(f"Successfully downloaded from Google Drive using Service Account")
                return True
            except ImportError:
                print("Google API libraries not installed. Install with:")
                print("  pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            except Exception as e:
                print(f"Error using Service Account credentials: {e}")
        
        # Method 2: Direct download with access token
        try:
            import requests
            
            if access_token:
                url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
                headers = {"Authorization": f"Bearer {access_token}"}
                response = requests.get(url, headers=headers, stream=True)
                response.raise_for_status()
                
                with open(output_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                return True
        except ImportError:
            print("requests not installed. Install with: pip install requests")
        except Exception as e:
            print(f"Error downloading with access token: {e}")
        
        # Method 3: Using gdown (if installed)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            return True
        except ImportError:
            print("gdown not installed. Install with: pip install gdown")
        except Exception as e:
            print(f"Error using gdown: {e}")
        
        return False
            
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False

def download_from_url(url, output_path, token=None):
    """Download file from a URL (with optional authentication)."""
    try:
        import requests
        
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        
        response = requests.get(url, headers=headers, stream=True)
        response.raise_for_status()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading from URL: {e}")
        return False

def download_from_s3(bucket, key, output_path, access_key=None, secret_key=None):
    """Download file from S3."""
    try:
        import boto3
        
        if access_key and secret_key:
            s3 = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key
            )
        else:
            # Use default credentials
            s3 = boto3.client('s3')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        s3.download_file(bucket, key, output_path)
        print(f"Downloaded {output_path} from S3")
        return True
    except Exception as e:
        print(f"Error downloading from S3: {e}")
        return False

def main():
    """Main function to download private test data."""
    # Get method from environment variable
    method = os.getenv('PRIVATE_DATA_METHOD', 'url').lower()
    output_path = Path('data/private/test.parquet')
    
    print(f"Downloading private test data using method: {method}")
    
    success = False
    
    if method == 'google_drive':
        file_id = os.getenv('GOOGLE_DRIVE_FILE_ID')
        access_token = os.getenv('GOOGLE_DRIVE_ACCESS_TOKEN')
        credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        
        if not file_id:
            print("ERROR: GOOGLE_DRIVE_FILE_ID environment variable not set")
            sys.exit(1)
        
        success = download_from_google_drive(file_id, str(output_path), access_token, credentials_json)
    
    elif method == 'url':
        url = os.getenv('PRIVATE_DATA_URL')
        token = os.getenv('PRIVATE_DATA_TOKEN')
        
        if not url:
            print("ERROR: PRIVATE_DATA_URL environment variable not set")
            sys.exit(1)
        
        success = download_from_url(url, str(output_path), token)
    
    elif method == 's3':
        bucket = os.getenv('S3_BUCKET')
        key = os.getenv('S3_KEY', 'data/private/test.parquet')
        access_key = os.getenv('AWS_ACCESS_KEY_ID')
        secret_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if not bucket:
            print("ERROR: S3_BUCKET environment variable not set")
            sys.exit(1)
        
        success = download_from_s3(bucket, key, str(output_path), access_key, secret_key)
    
    else:
        print(f"Unknown method: {method}")
        print("Supported methods: google_drive, url, s3")
        sys.exit(1)
    
    if not success:
        print("ERROR: Failed to download private test data")
        print("Please configure GitHub Secrets for private data access")
        sys.exit(1)
    
    # Verify file exists and has content
    if not output_path.exists() or output_path.stat().st_size == 0:
        print("ERROR: Downloaded file is empty or doesn't exist")
        print("Please check your private data configuration")
        sys.exit(1)
    
    print(f"Successfully downloaded {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()

