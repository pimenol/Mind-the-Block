# Team guide

This page tells you which files to read and how to add your work without
conflicts with other people.

## Read these 5 files, in this order (about 20 minutes)

1. `research/HYPOTHESES.md`: the questions and the result that proves each one wrong.
2. `research/common.py`: shared helpers. `load` gives the data, `make_split` gives the windows, `get_model` trains or loads a GRU.
3. `research/e1_age_stratified.py`: one complete experiment as an example.
4. `results/`: one JSON file per experiment. Every number in the report comes from here.
5. `report/report.html`: the full write-up. Open it in a browser.

You can ignore the rest. `starter.py`, `check_data.py`, `generate_data.py` and
`data/` come from the course and nobody edits them.

## Where your code goes

| Folder | What is in it | Who edits it |
|---|---|---|
| `research/scratch/<your_name>.py` | your work for this week | only you |
| `research/common.py` | shared helpers | everyone, in small separate commits |
| `research/eN_<topic>.py` | finished experiments | the author, after the team agrees |
| `results/` | JSON output of the scripts | the scripts, never by hand |
| `report/` | the report and its build script | one person per week |

To start, copy the template and run it from the repo root:

```bash
cp research/scratch/template.py research/scratch/anna.py
python research/scratch/anna.py          # writes results/scratch_anna.json
```

When your result is final, rename the file to the next `research/eN_<topic>.py`
and add its hypothesis to `research/HYPOTHESES.md` before you run it again.

## How to commit

You only edit your own file, so the commands are always the same:

```bash
git pull --rebase
git add research/scratch/anna.py results/scratch_anna.json
git commit -m "feat: test age weighting on block windows"
git push
```

Rules that prevent conflicts:

1. Add files by name. Do not use `git add -A`.
2. If you change `research/common.py`, commit only that change, and tell the team in the chat.
3. Do not edit `report/report.html` or `results/*.json` by hand. If git shows a conflict in them, run the script again (`python report/build_report.py`) and commit the new file.
4. Commit messages are one line: `<type>: <description>`. Types: `feat`, `fix`, `docs`, `data`, `refactor`, `chore`, `test`.
