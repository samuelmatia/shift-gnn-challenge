"""
Update leaderboard from scores file.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Load existing leaderboard
leaderboard_file = Path("leaderboard.json")
if leaderboard_file.exists():
    with open(leaderboard_file, 'r') as f:
        leaderboard = json.load(f)
else:
    leaderboard = {"last_updated": None, "submissions": []}

# Parse scores
scores_file = Path("results/scores.txt")
if not scores_file.exists():
    print("No scores file found")
    sys.exit(1)

# Check if file is empty
if scores_file.stat().st_size == 0:
    print("Scores file is empty")
    sys.exit(1)

existing_map = {sub['team']: sub for sub in leaderboard.get('submissions', [])}

with open(scores_file, 'r') as f:
    for line in f:
        parts = line.strip().split(':')
        if len(parts) >= 4:
            team = parts[0]
            weighted_f1 = float(parts[1])
            overall_f1 = float(parts[2])
            rare_f1 = float(parts[3])
            
            entry = {
                'team': team,
                'submission_file': f"submissions/{team}.csv",
                'weighted_f1': weighted_f1,
                'overall_f1': overall_f1,
                'rare_f1': rare_f1,
                'timestamp': datetime.now().isoformat()
            }
            
            # Update if better score or new team
            if team not in existing_map or entry['weighted_f1'] > existing_map[team]['weighted_f1']:
                existing_map[team] = entry
                print(f"Updated entry for {team}: {weighted_f1:.6f}")

# Quick filter: only check files for entries we're keeping
# Note: In PR context, we keep all entries from main even if files don't exist in PR branch
# The files will exist in main after merge
submissions_dir = Path("submissions")
valid_submissions = []

for entry in existing_map.values():
    # Check if the submission file exists (quick check)
    team = entry['team']
    submission_file = submissions_dir / f"{team}.csv"
    
    if submission_file.exists():
        valid_submissions.append(entry)
    else:
        # In PR context, don't remove entries - files exist in main
        # Only remove if we're sure the file doesn't exist (e.g., after explicit deletion)
        # For now, keep all entries to preserve leaderboard during PR evaluation
        valid_submissions.append(entry)
        print(f"Note: File not found for {team} in current branch, but keeping in leaderboard (may be in main)")

# Sort and save
valid_submissions.sort(key=lambda x: x['weighted_f1'], reverse=True)

leaderboard = {
    'last_updated': datetime.now().isoformat(),
    'submissions': valid_submissions
}

with open(leaderboard_file, 'w') as f:
    json.dump(leaderboard, f, indent=2)

print(f"Leaderboard updated with {len(valid_submissions)} teams")

