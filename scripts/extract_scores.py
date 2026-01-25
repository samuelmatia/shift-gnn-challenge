"""
Extract scores from evaluation output and format them for leaderboard.
"""
import json
import sys
import re
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python extract_scores.py <team_name>", file=sys.stderr)
    sys.exit(1)

team_name = sys.argv[1]
output_file = Path(f"results/{team_name}_output.txt")

if not output_file.exists():
    print(f"{team_name}:0.0:0.0:0.0", file=sys.stderr)
    sys.exit(1)

try:
    with open(output_file, 'r') as f:
        content = f.read()
    
    # Find JSON in output
    json_match = re.search(r'\{.*\}', content, re.DOTALL)
    if json_match:
        scores = json.loads(json_match.group())
        print(f"{team_name}:{scores.get('weighted_f1', 0.0):.6f}:{scores.get('overall_macro_f1', 0.0):.6f}:{scores.get('rare_transitions_f1', 0.0):.6f}")
    else:
        print(f"{team_name}:0.0:0.0:0.0", file=sys.stderr)
except Exception as e:
    print(f"Error processing {team_name}: {e}", file=sys.stderr)
    print(f"{team_name}:0.0:0.0:0.0", file=sys.stderr)

