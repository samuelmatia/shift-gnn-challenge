"""
Remove a submission from the leaderboard and optionally delete the submission file.
Usage: python scripts/remove_submission.py <team_name> [--delete-file]
"""
import json
import sys
import argparse
from pathlib import Path
from datetime import datetime

def remove_submission(team_name, delete_file=False):
    """
    Remove a submission from the leaderboard.
    
    Args:
        team_name: Name of the team to remove (without .csv extension)
        delete_file: If True, also delete the submission file
    """
    leaderboard_file = Path("leaderboard.json")
    
    # Load existing leaderboard
    if not leaderboard_file.exists():
        print(f"ERROR: {leaderboard_file} not found!")
        sys.exit(1)
    
    with open(leaderboard_file, 'r') as f:
        leaderboard = json.load(f)
    
    # Find and remove the submission
    submissions = leaderboard.get('submissions', [])
    original_count = len(submissions)
    
    # Filter out the team
    leaderboard['submissions'] = [
        sub for sub in submissions 
        if sub.get('team', '').lower() != team_name.lower()
    ]
    
    removed_count = original_count - len(leaderboard['submissions'])
    
    if removed_count == 0:
        print(f"WARNING: Team '{team_name}' not found in leaderboard.")
        print(f"Available teams: {[sub.get('team') for sub in submissions]}")
        sys.exit(1)
    
    # Update timestamp
    leaderboard['last_updated'] = datetime.now().isoformat()
    
    # Save updated leaderboard
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    print(f"✅ Removed '{team_name}' from leaderboard.")
    print(f"   Remaining teams: {len(leaderboard['submissions'])}")
    
    # Optionally delete the submission file
    if delete_file:
        submission_file = Path("submissions") / f"{team_name}.csv"
        if submission_file.exists():
            submission_file.unlink()
            print(f"✅ Deleted submission file: {submission_file}")
        else:
            print(f"⚠️  Submission file not found: {submission_file}")
    
    # Regenerate HTML leaderboard
    try:
        import sys
        script_dir = Path(__file__).parent
        sys.path.insert(0, str(script_dir))
        from generate_leaderboard import generate_html
        generate_html(leaderboard)
        print("✅ Regenerated leaderboard.html")
    except Exception as e:
        print(f"⚠️  Could not regenerate HTML: {e}")
        print("   You may need to run: python scripts/generate_leaderboard.py")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Remove a submission from the leaderboard'
    )
    parser.add_argument(
        'team_name',
        help='Name of the team to remove (without .csv extension)'
    )
    parser.add_argument(
        '--delete-file',
        action='store_true',
        help='Also delete the submission CSV file'
    )
    
    args = parser.parse_args()
    
    remove_submission(args.team_name, args.delete_file)

