"""
Script to update the leaderboard with new submission results.
"""
import json
import argparse
from datetime import datetime
from pathlib import Path

def load_leaderboard():
    """Load existing leaderboard or create new one."""
    leaderboard_file = Path("leaderboard.json")
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            return json.load(f)
    else:
        return {
            "last_updated": None,
            "submissions": []
        }

def save_leaderboard(leaderboard):
    """Save leaderboard to JSON file."""
    leaderboard["last_updated"] = datetime.now().isoformat()
    with open("leaderboard.json", 'w') as f:
        json.dump(leaderboard, f, indent=2)

def update_leaderboard(submission_file, team_name, weighted_f1, overall_f1, rare_f1):
    """Update leaderboard with new submission."""
    leaderboard = load_leaderboard()
    
    # Create new entry
    entry = {
        "team": team_name,
        "submission_file": submission_file,
        "weighted_f1": float(weighted_f1),
        "overall_f1": float(overall_f1),
        "rare_f1": float(rare_f1),
        "timestamp": datetime.now().isoformat()
    }
    
    # Check if team already exists
    existing_idx = None
    for i, sub in enumerate(leaderboard["submissions"]):
        if sub["team"] == team_name:
            existing_idx = i
            break
    
    if existing_idx is not None:
        # Update existing entry if this score is better
        if entry["weighted_f1"] > leaderboard["submissions"][existing_idx]["weighted_f1"]:
            leaderboard["submissions"][existing_idx] = entry
            print(f"Updated entry for {team_name} with better score: {weighted_f1}")
        else:
            print(f"Existing entry for {team_name} has better score. Keeping existing.")
    else:
        # Add new entry
        leaderboard["submissions"].append(entry)
        print(f"Added new entry for {team_name}")
    
    # Sort by weighted_f1 (descending)
    leaderboard["submissions"].sort(key=lambda x: x["weighted_f1"], reverse=True)
    
    # Save to JSON
    save_leaderboard(leaderboard)
    
    # Generate HTML
    generate_html_leaderboard(leaderboard)
    
    return leaderboard

def generate_html_leaderboard(leaderboard):
    """Generate HTML leaderboard page."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Challenge - Leaderboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #7f8c8d;
            margin-bottom: 30px;
        }
        .last-updated {
            text-align: center;
            color: #95a5a6;
            font-size: 0.9em;
            margin-bottom: 30px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-radius: 8px;
            overflow: hidden;
        }
        thead {
            background: #3498db;
            color: white;
        }
        th {
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        td {
            padding: 12px 15px;
            border-bottom: 1px solid #ecf0f1;
        }
        tbody tr:hover {
            background: #f8f9fa;
        }
        tbody tr:last-child td {
            border-bottom: none;
        }
        .rank {
            font-weight: bold;
            color: #2c3e50;
            width: 60px;
        }
        .rank-1 { color: #f39c12; font-size: 1.2em; }
        .rank-2 { color: #95a5a6; font-size: 1.1em; }
        .rank-3 { color: #e67e22; font-size: 1.05em; }
        .team-name {
            font-weight: 600;
            color: #2c3e50;
        }
        .score {
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        .primary-score {
            color: #27ae60;
            font-size: 1.1em;
        }
        .medal {
            font-size: 1.5em;
        }
        .empty {
            text-align: center;
            padding: 40px;
            color: #95a5a6;
        }
    </style>
</head>
<body>
    <h1>🏆 GNN Challenge Leaderboard</h1>
    <p class="subtitle">Role Transition Prediction in Temporal Networks</p>
    <p class="last-updated">Last updated: """ + (leaderboard.get("last_updated", "Never") or "Never") + """</p>
    
    <table>
        <thead>
            <tr>
                <th class="rank">Rank</th>
                <th>Team</th>
                <th class="score primary-score">Weighted Macro-F1</th>
                <th class="score">Overall Macro-F1</th>
                <th class="score">Rare Transitions F1</th>
                <th>Submission Time</th>
            </tr>
        </thead>
        <tbody>
"""
    
    if not leaderboard.get("submissions"):
        html += """            <tr>
                <td colspan="6" class="empty">No submissions yet. Be the first!</td>
            </tr>
"""
    else:
        for idx, entry in enumerate(leaderboard["submissions"], 1):
            rank_class = f"rank-{idx}" if idx <= 3 else ""
            medal = ""
            if idx == 1:
                medal = "🥇 "
            elif idx == 2:
                medal = "🥈 "
            elif idx == 3:
                medal = "🥉 "
            
            timestamp = entry.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
                except:
                    pass
            
            html += f"""            <tr>
                <td class="rank {rank_class}">{medal}{idx}</td>
                <td class="team-name">{entry['team']}</td>
                <td class="score primary-score">{entry['weighted_f1']:.6f}</td>
                <td class="score">{entry['overall_f1']:.6f}</td>
                <td class="score">{entry['rare_f1']:.6f}</td>
                <td>{timestamp}</td>
            </tr>
"""
    
    html += """        </tbody>
    </table>
    
    <p style="text-align: center; margin-top: 30px; color: #95a5a6; font-size: 0.9em;">
        Submit your solution via Pull Request to appear on the leaderboard!
    </p>
</body>
</html>"""
    
    with open("leaderboard.html", 'w') as f:
        f.write(html)
    
    print("Generated leaderboard.html")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update leaderboard with submission results")
    parser.add_argument("--submission", required=True, help="Path to submission file")
    parser.add_argument("--team", required=True, help="Team name")
    parser.add_argument("--weighted-f1", required=True, type=float, help="Weighted Macro-F1 score")
    parser.add_argument("--overall-f1", required=True, type=float, help="Overall Macro-F1 score")
    parser.add_argument("--rare-f1", required=True, type=float, help="Rare Transitions F1 score")
    
    args = parser.parse_args()
    
    update_leaderboard(
        args.submission,
        args.team,
        args.weighted_f1,
        args.overall_f1,
        args.rare_f1
    )

