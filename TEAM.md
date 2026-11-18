# Team guide

`main` holds the course starter and a small shared kit. Each person works on
Step 1 with their own hypotheses in their own branch. Nobody commits
experiments directly to `main`.

## What is in `main`

| File | What it is |
|---|---|
| `README.md` | the course task. Read it first. |
| `starter.py`, `check_data.py`, `generate_data.py`, `data/` | the course starter. Nobody edits these files. |
| `research/common.py` | shared helpers: `load` gives the data, `make_split` gives the windows, `get_model` trains or loads a GRU |
| `research/template.py` | a complete experiment in 25 lines. Copy it for each new experiment. |
| `research/HYPOTHESES.md` | an empty table for your hypotheses and their falsifiers |

## Start your branch (once)

```bash
git clone git@github.com:pimenol/Mind-the-Block.git
cd Mind-the-Block
git checkout -b anna            # use your name
pip install -r requirements.txt
python starter.py --quick       # makes sure that the starter runs
```

## Do an experiment

1. Write the hypothesis and its falsifier in `research/HYPOTHESES.md`.
2. Copy the template: `cp research/template.py research/e1_<topic>.py`.
3. Edit the copy and run it from the repo root: `python research/e1_<topic>.py`. It writes `results/e1_<topic>.json`.

## Commit and push

```bash
git add research/HYPOTHESES.md research/e1_<topic>.py results/e1_<topic>.json
git commit -m "feat: test stale sensor-0 hypothesis"
git push -u origin anna         # after the first time, git push is enough
```

Add files by name. Do not use `git add -A`. Commit messages are one line:
`<type>: <description>`. Types: `feat`, `fix`, `docs`, `data`, `refactor`,
`chore`, `test`.

## Change a shared helper

Other people can need your change in `research/common.py` too. Make it in a
separate commit and open a pull request to `main`. When the pull
request is merged, everyone gets it with one command in their own branch:

```bash
git pull origin main
```

Do not edit `results/*.json` by hand. The scripts write them.
