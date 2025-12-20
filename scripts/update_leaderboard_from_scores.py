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

# Sort and save
submissions = list(existing_map.values())
submissions.sort(key=lambda x: x['weighted_f1'], reverse=True)

leaderboard = {
    'last_updated': datetime.now().isoformat(),
    'submissions': submissions
}

with open(leaderboard_file, 'w') as f:
    json.dump(leaderboard, f, indent=2)

print(f"Leaderboard updated with {len(submissions)} teams")

