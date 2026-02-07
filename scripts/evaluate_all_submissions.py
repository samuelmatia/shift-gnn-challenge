"""
Evaluate submissions in the submissions/ directory only.
submissions_examples/ is never read (examples only, no evaluation, no leaderboard).
In PR context: only submissions/challenge_submission.csv is considered (fixed filename).
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
    # Only submissions/ is evaluated; submissions_examples/ is ignored
    submissions_dir = Path(__file__).parent.parent / 'submissions'
    results = []
    # PR context: one submission per PR, team name = GitHub username (set by workflow)
    pr_team_name = os.environ.get('PR_TEAM_NAME', '').strip()

    # PR context: only challenge_submission.csv is considered (fixed filename)
    REQUIRED_PR_FILENAME = 'challenge_submission.csv'
    metadata = {'model_type': 'unknown', 'notes': ''}

    if pr_team_name:
        submission_file = submissions_dir / REQUIRED_PR_FILENAME
        if not submission_file.exists():
            print(f"PR mode: {REQUIRED_PR_FILENAME} not found in submissions/")
            print(f"Please add or rename your file to submissions/{REQUIRED_PR_FILENAME}")
            results_file = Path(__file__).parent.parent / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump([], f, indent=2)
            return
        # Load metadata.json (model_type: human | llm | human+llm, notes: optional)
        metadata = {'model_type': 'unknown', 'notes': ''}
        metadata_file = submissions_dir / 'metadata.json'
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except json.JSONDecodeError:
                pass
        csv_files = [submission_file]
        print(f"PR mode: evaluating {REQUIRED_PR_FILENAME} (team = {pr_team_name}, model_type = {metadata.get('model_type', 'unknown')})")
    else:
        # Main branch: evaluate all CSV files in submissions/
        csv_files = list(submissions_dir.glob('*.csv'))
        csv_files = [f for f in csv_files if 'sample' not in f.name.lower()]
        csv_files.sort(key=lambda f: f.name)
        if not csv_files:
            print("No submission files found in submissions/ directory")
            results_file = Path(__file__).parent.parent / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump([], f, indent=2)
            return
        print(f"Found {len(csv_files)} submission(s) to evaluate")

    for csv_file in csv_files:
        print(f"\nEvaluating {csv_file.name}...")
        scores = evaluate_submission(str(csv_file))

        if scores:
            team_name = pr_team_name if pr_team_name else csv_file.stem
            entry = {'file': str(csv_file), 'team': team_name, 'scores': scores}
            if pr_team_name:
                entry['model_type'] = metadata.get('model_type', 'unknown')
                entry['notes'] = metadata.get('notes', '')
            results.append(entry)
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

