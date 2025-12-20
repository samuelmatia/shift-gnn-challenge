"""
Script to download private test data from a secure location.
This script is used by GitHub Actions to access test labels.
"""
import os
import sys
from pathlib import Path

def download_from_google_drive(file_id, output_path, access_token=None):
    """Download file from Google Drive using file ID."""
    try:
        import requests
        
        # Method 1: Direct download with access token
        if access_token:
            url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media"
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
        
        # Method 2: Using gdown (if installed)
        try:
            import gdown
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, output_path, quiet=False)
            return True
        except ImportError:
            print("gdown not installed. Install with: pip install gdown")
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
        
        if not file_id:
            print("ERROR: GOOGLE_DRIVE_FILE_ID environment variable not set")
            sys.exit(1)
        
        success = download_from_google_drive(file_id, str(output_path), access_token)
    
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
        sys.exit(1)
    
    # Verify file exists and has content
    if not output_path.exists() or output_path.stat().st_size == 0:
        print("ERROR: Downloaded file is empty or doesn't exist")
        sys.exit(1)
    
    print(f"Successfully downloaded {output_path}")
    print(f"File size: {output_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()

