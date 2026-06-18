# Redrob AI Candidate Ranker

Intelligent candidate ranking system for Redrob Hackathon.

## Approach
- BGE-small-en-v1.5 semantic embeddings (pre-computed offline)
- Multi-signal scoring: title relevance, career evidence, skills with trust multiplier, experience, location, education, GitHub activity
- Behavioral signals modifier (response rate, activity, notice period)
- Honeypot detection (22K+ flagged)

## Setup
```bash
pip install sentence-transformers numpy
python3 precompute.py   # run once - generates embeddings
python3 rank.py         # generates submission.csv
python3 validate_submission.py submission.csv
```

## Compute
- Runtime: ~3 min on CPU (MacBook Air M2)
- Memory: ~2 GB
- No network during ranking
