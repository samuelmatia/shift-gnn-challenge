"""
Generate leaderboard from evaluation results.
"""
import json
from datetime import datetime
from pathlib import Path

def load_evaluation_results():
    """Load evaluation results."""
    results_file = Path(__file__).parent.parent / 'evaluation_results.json'
    if results_file.exists():
        with open(results_file, 'r') as f:
            return json.load(f)
    return []

def load_existing_leaderboard():
    """Load existing leaderboard."""
    leaderboard_file = Path(__file__).parent.parent / 'leaderboard.json'
    if leaderboard_file.exists():
        with open(leaderboard_file, 'r') as f:
            return json.load(f)
    return {"last_updated": None, "submissions": []}

def generate_leaderboard():
    """Generate leaderboard from all results."""
    results = load_evaluation_results()
    existing = load_existing_leaderboard()
    
    # Create a map of existing submissions
    existing_map = {sub['team']: sub for sub in existing.get('submissions', [])}
    
    # Process new results
    for result in results:
        team = result['team']
        scores = result['scores']
        
        entry = {
            'team': team,
            'submission_file': result['file'],
            'weighted_f1': scores.get('weighted_f1', 0.0),
            'overall_f1': scores.get('overall_macro_f1', 0.0),
            'rare_f1': scores.get('rare_transitions_f1', 0.0),
            'timestamp': datetime.now().isoformat()
        }
        
        # Update if better score or new team
        if team not in existing_map or entry['weighted_f1'] > existing_map[team]['weighted_f1']:
            existing_map[team] = entry
    
    # Convert to list and sort
    submissions = list(existing_map.values())
    submissions.sort(key=lambda x: x['weighted_f1'], reverse=True)
    
    leaderboard = {
        'last_updated': datetime.now().isoformat(),
        'submissions': submissions
    }
    
    # Save JSON
    leaderboard_file = Path(__file__).parent.parent / 'leaderboard.json'
    with open(leaderboard_file, 'w') as f:
        json.dump(leaderboard, f, indent=2)
    
    # Generate HTML
    generate_html(leaderboard)
    
    print(f"Generated leaderboard with {len(submissions)} teams")
    return leaderboard

def generate_html(leaderboard):
    """Generate HTML leaderboard."""
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GNN Challenge - Leaderboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }
        .last-updated {
            text-align: center;
            color: white;
            margin-bottom: 20px;
            opacity: 0.8;
        }
        .leaderboard {
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        thead {
            background: #2c3e50;
            color: white;
        }
        th {
            padding: 18px;
            text-align: left;
            font-weight: 600;
            font-size: 0.95em;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        tbody tr {
            border-bottom: 1px solid #ecf0f1;
            transition: background 0.2s;
        }
        tbody tr:hover {
            background: #f8f9fa;
        }
        tbody tr:last-child {
            border-bottom: none;
        }
        td {
            padding: 15px 18px;
        }
        .rank {
            font-weight: bold;
            font-size: 1.1em;
            width: 80px;
        }
        .rank-1 { color: #f39c12; }
        .rank-2 { color: #95a5a6; }
        .rank-3 { color: #e67e22; }
        .medal {
            font-size: 1.3em;
            margin-right: 5px;
        }
        .team-name {
            font-weight: 600;
            color: #2c3e50;
            font-size: 1.05em;
        }
        .score {
            font-family: 'Courier New', monospace;
            font-weight: bold;
        }
        .primary-score {
            color: #27ae60;
            font-size: 1.15em;
        }
        .empty {
            text-align: center;
            padding: 60px;
            color: #95a5a6;
            font-size: 1.1em;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            color: white;
            opacity: 0.9;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏆 GNN Challenge Leaderboard</h1>
            <p>Role Transition Prediction in Temporal Networks</p>
        </div>
        <p class="last-updated">Last updated: """ + (leaderboard.get("last_updated", "Never") or "Never") + """</p>
        
        <div class="leaderboard">
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
        html += """                    <tr>
                        <td colspan="6" class="empty">No submissions yet. Be the first! 🚀</td>
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
                    timestamp = dt.strftime("%Y-%m-%d %H:%M")
                except:
                    pass
            
            html += f"""                    <tr>
                        <td class="rank {rank_class}"><span class="medal">{medal}</span>{idx}</td>
                        <td class="team-name">{entry['team']}</td>
                        <td class="score primary-score">{entry['weighted_f1']:.6f}</td>
                        <td class="score">{entry['overall_f1']:.6f}</td>
                        <td class="score">{entry['rare_f1']:.6f}</td>
                        <td>{timestamp}</td>
                    </tr>
"""
    
    html += """                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Submit your solution via Pull Request to appear on the leaderboard!</p>
        </div>
    </div>
</body>
</html>"""
    
    html_file = Path(__file__).parent.parent / 'leaderboard.html'
    with open(html_file, 'w') as f:
        f.write(html)
    
    print(f"Generated {html_file}")

if __name__ == '__main__':
    generate_leaderboard()

