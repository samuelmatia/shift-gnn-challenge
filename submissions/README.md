# Submissions (Pull Request)

**Required files:**
- `challenge_submission.csv` — your predictions
- `metadata.json` — records whether the submission was produced by a **human**, an **LLM**, or **both**

**metadata.json format:** (see `metadata.json.example`)
```json
{"model_type": "human", "notes": "Optional notes"}
```
`model_type` must be one of: `human`, `llm`, `human+llm`. If missing, defaults to `unknown`.

- Your **leaderboard name** is your **GitHub username** (one entry per participant).
- You can update your submission by pushing new commits.
- Do **not** put your file in `submissions_examples/` — that folder is for examples only and is never evaluated.

Format for CSV: columns `user_id`, `snapshot_id`, `predicted_role` (see main README).
