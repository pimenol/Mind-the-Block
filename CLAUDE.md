# Project 04: forecasting with missing readings

This file gives guidance to Claude Code (claude.ai/code) for work in this
repository. `README.md` states the course task. `TEAM.md` explains the branch
workflow.

## Hard rules from the assignment

- Never run `python starter.py --final` or read the `test_*` arrays until the
  claim, method and settings are frozen. Never tune on final results.
  `research/common.load` returns only `train_*` and `dev_*`.
- Targets are for loss and evaluation only. Never use a target or a future
  value to fill an input.
- Mean and scale come from training observations only.
- Do not edit `starter.py`, `check_data.py`, `generate_data.py` or
  `MANIFEST_SHA256.json`. New code goes in `research/`.
- The replication unit is the data seed (940, 941, 942). Report per-seed
  values, not only a mean.

## Layout

- `starter.py`, `check_data.py`, `generate_data.py`, `data/`: the course starter.
- `research/common.py`: shared helpers (`load`, `make_split`, `get_model`, `save`).
- `research/template.py`: a minimal experiment. Copy it to `research/eN_<topic>.py`.
- `research/HYPOTHESES.md`: hypotheses and falsifiers, written before each run.
- `results/`: one JSON file per experiment. `results/models/` is a GRU cache, not in git.

## Commands

```bash
pip install -r requirements.txt
python check_data.py                 # data and GRU gradient checks
python starter.py --quick            # smoke run
python research/template.py          # example experiment, writes results/template.json
```

Commit messages are one line: `<type>: <description>`. Add files by name.
