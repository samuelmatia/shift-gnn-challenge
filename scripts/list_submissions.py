"""
List all submissions in the leaderboard.
Usage: python scripts/list_submissions.py
"""
import json
from pathlib import Path

def list_submissions():
    """List all submissions in the leaderboard."""
    leaderboard_file = Path("leaderboard.json")
    
    if not leaderboard_file.exists():
        print("ERROR: leaderboard.json not found!")
        return
    
    with open(leaderboard_file, 'r') as f:
        leaderboard = json.load(f)
    
    submissions = leaderboard.get('submissions', [])
    
    if not submissions:
        print("No submissions in leaderboard.")
        return
    
    print(f"\n📊 Leaderboard ({len(submissions)} teams):\n")
    print(f"{'Rank':<6} {'Team Name':<40} {'Weighted F1':<15} {'File Exists':<12}")
    print("-" * 80)
    
    submissions_dir = Path("submissions")
    
    for idx, sub in enumerate(submissions, 1):
        team = sub.get('team', 'Unknown')
        weighted_f1 = sub.get('weighted_f1', 0.0)
        file_path = submissions_dir / f"{team}.csv"
        file_exists = "✅ Yes" if file_path.exists() else "❌ No"
        
        print(f"{idx:<6} {team:<40} {weighted_f1:<15.6f} {file_exists:<12}")
    
    print("\n" + "=" * 80)
    print(f"Last updated: {leaderboard.get('last_updated', 'Never')}")

if __name__ == '__main__':
    list_submissions()

