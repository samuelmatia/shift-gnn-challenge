"""
Evaluate all submissions in the submissions/ directory.
"""
import os
import json
import subprocess
from pathlib import Path

def evaluate_submission(submission_file):
    """Evaluate a single submission and return scores."""
    try:
        # Run scoring script with JSON output
        script_dir = Path(__file__).parent.parent
        result = subprocess.run(
            ['python', str(script_dir / 'scoring_script.py'), submission_file, '--json'],
            capture_output=True,
            text=True,
            cwd=script_dir
        )
        
        if result.returncode != 0:
            print(f"Error evaluating {submission_file}:")
            print(result.stderr)
            return None
        
        # Extract JSON from output
        output = result.stdout
        json_start = output.find('{')
        if json_start == -1:
            print(f"Could not find JSON in output for {submission_file}")
            return None
        
        json_str = output[json_start:]
        json_end = json_str.rfind('}')
        if json_end != -1:
            json_str = json_str[:json_end+1]
        
        try:
            scores = json.loads(json_str)
            return scores
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for {submission_file}: {e}")
            return None
            
    except Exception as e:
        print(f"Exception evaluating {submission_file}: {e}")
        return None

def main():
    submissions_dir = Path(__file__).parent.parent / 'submissions'
    results = []
    
    # Find all CSV files (except sample submissions)
    csv_files = list(submissions_dir.glob('*.csv'))
    csv_files = [f for f in csv_files if 'sample' not in f.name.lower()]
    
    if not csv_files:
        print("No submission files found in submissions/ directory")
        # Create empty results file so generate_leaderboard doesn't fail
        results_file = Path(__file__).parent.parent / 'evaluation_results.json'
        with open(results_file, 'w') as f:
            json.dump([], f, indent=2)
        return
    
    print(f"Found {len(csv_files)} submission(s) to evaluate")
    
    for csv_file in csv_files:
        print(f"\nEvaluating {csv_file.name}...")
        scores = evaluate_submission(str(csv_file))
        
        if scores:
            team_name = csv_file.stem
            results.append({
                'file': str(csv_file),
                'team': team_name,
                'scores': scores
            })
            print(f"✓ Successfully evaluated {csv_file.name}")
        else:
            print(f"✗ Failed to evaluate {csv_file.name}")
    
    # Save results
    results_file = Path(__file__).parent.parent / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved evaluation results to {results_file}")

if __name__ == '__main__':
    main()

