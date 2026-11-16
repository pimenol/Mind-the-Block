# Enzyme Literature Mining — GARDEN

AI agent(s) for automatically extracting enzyme data from scientific literature.
The project is named **GARDEN**: a Growing Agentic Resource for Data-mining of ENzymes.
The pipeline is **enzyme-class agnostic** — it works on any enzyme family, not just
terpene synthases. Terpene synthases (via MARTS-DB) are the first benchmark target.

## Design Philosophy

The dataset this pipeline produces is about **real enzymes + real reactions**,
recorded as machine-readable amino-acid sequences and SMILES.

A reaction entry (`activity: "active"`) is *valid* iff it carries:

1. a real **amino-acid `sequence`** for every chain in `subunits` — the enzyme
   variant used in the paper, and
2. **`substrates` / `products` SMILES** for a reaction the paper performed
   experimentally (in vitro assay, in vivo expression, GC-/LC-MS product
   identification, crystal structure of an enzyme-substrate complex, …), and
3. a **mass- and charge-balanced** equation — substrate-side and product-side
   atoms (incl. hydrogens) and net charge match.

Following Rhea, every reaction is a *complete, balanced transformation* and there
is **no substrate/cofactor distinction** among the net-transformed participants:
consumed/produced cofactors (NADPH/NADP⁺, ATP/ADP, …) live in `substrates` /
`products` on their correct side, and only **required-but-not-consumed** catalytic
cofactors (active-site metals, regenerated prosthetic groups) go in the separate
`cofactors` field (UniProt's `Cofactor` layer, outside the equation). The
*transformation* must be paper-demonstrated, but its *stoichiometry is completed*
to balance by chemical necessity — never by inventing molecules.

**Broken rows are repaired, not just flagged.** Right after the agent writes its
rows, the runner checks every row with the shared validator
(`src/extract/validate.py`: SMILES present and readable, both sides of a
reaction, a sequence of ≥ 50 residues for every chain, the equation balanced,
names lined up with SMILES, usable numbers and units). When something is wrong it
resumes the same agent session with the exact problem list and asks for a
corrected file, for up to two repair rounds (`--max-repair-rounds`). Only a
problem that survives — the paper and the tools really can't resolve it — ends up
as `agent-needs-human`; the aggregator repeats the same checks as a safety net
(`validation: …` notes) and flags, never drops.

**Negative results** are rows too (`activity: "inactive"`): a combination the paper
experimentally tested and found to have **no detectable** activity — a dead mutant,
a substrate the enzyme does not convert, a metal that does not support activity, a
candidate enzyme inactive with everything tested. Such a row records what was tested
in the existing fields (`subunits` = the chain(s) assayed, assayed `substrates`,
`cofactors` present) with `products: []` — nothing was formed, so nothing to balance; the
evidence and any stated cause go in `catalytic_justification`. Any detected product,
even trace, makes the combination active (a reaction row when the product is
identified, else no row). There is deliberately no sub-type field: the cause of a
negative is open-ended and often unknowable. Negatives share
`extracted_reactions.json` with the reactions, discriminated by `activity`; the
benchmark scorer never scores them.

Hypothetical or predicted-but-untested reactions are **out of scope** — if a paper
merely speculates that an enzyme "might also accept X" without assaying it, that
reaction does not get a row (neither a reaction nor a negative result). Every other
field (names, EC, yield, assay, justifications, notes) is metadata that helps
interpret, audit, or deduplicate the load-bearing `(subunits sequences, SMILES
tuple)` and its `activity` label.

**Empty values:** a missing or unknown value is `[]` for a list and `""` for
everything else, numbers included — never `null` (the aggregator rewrites any
`null`, including in old runs).

## Project Goal

Mine structured enzyme data from papers. One reaction (or negative result) = one
row. Each row has:

- **Enzyme block** — `enzyme_name`, `subunits`, `ec_number`,
  `enzyme_justification`. The enzyme is described **only** through `subunits`:
  one object per distinct polypeptide chain with exactly the keys `name`,
  `sequence`, `uniprot_id`, `genbank_id`, `catalytic_residues` (1-indexed
  positions into *that chain's* sequence for residues the paper assigns a
  catalytic role; `[]` when not reported), `copies` (copies of the chain in the
  active enzyme as the paper experimentally established it; `""` when not
  established) and `role` (`catalytic` / `regulatory` / `structural` /
  `electron_transfer` / `other` / `""`). A monomer or homo-oligomer is one object
  (`copies` 1 / n / `""`); a hetero-oligomer is several, the active-site chain
  first; one row per reaction, never per chain. How the composition was
  established goes in `enzyme_justification`. Rows from runs before v4 carry
  top-level `sequence` / `uniprot_id` / `genbank_id` / `catalytic_residues`
  instead; `src/utils/subunits.py:row_subunits` reads both formats (the
  aggregator converts them, the scorer / report read them as-is).
- **Activity** — `activity`: `"active"` (a reaction) or `"inactive"` (a negative
  result: tested, no detectable activity, `products: []`); legacy rows without it
  are reactions
- **Reaction names** — `substrate_names`, `product_names` (the balanced
  participants, including stoichiometric cofactors on their correct side) and
  `cofactor_names` (required-but-not-consumed catalytic cofactors only — metals,
  regenerated prosthetic groups); all lists, one entry per molecule
- **Reaction SMILES** (load-bearing) — `substrates`, `products`, `cofactors`
  (parallel lists of canonical isomeric SMILES, one per name; the reaction must
  mass/charge balance)
- **Reaction justification** — `reaction_justification`: audit trail for how
  the SMILES were derived (which figures/tables, which `name_to_smiles` calls,
  any consistency flags)
- **Catalytic context** — `kinetics` (list of `{parameter, value, unit, note}`),
  `product_abundance`, `product_fraction_pct`, `optimal_pH`,
  `optimal_temperature_c`, `thermostability` (list), `selectivity` (list of
  enums), `catalytic_justification`. The enzyme's measured performance; replaces
  the old terpene-centric `product_yield`. Values are unit-canonicalized via the
  `catalytic_parameters` MCP server (one canonical unit per parameter); see
  `assets/reports/REPORT_21Jun26_catalytic_params_survey.md`.
- **Context** — `species`, `assay`, `doi`, `notes`
- **Status** — `status` and `status_note` (see below)

### Row status: `status` + `status_note`

Every row carries a small review-status pair so curators can triage
agent-extracted output without reading the prose justifications.

- `status` is one of three string values:
  - `"agent"` — agent extracted, every load-bearing field is populated, no
    fallback heuristic fired with a result the agent had to reject. The
    default for clean rows.
  - `"agent-needs-human"` — agent extracted but flagged at least one issue
    a curator should look at (missing SMILES, tentative product ID,
    ambiguous identifier, OCR fallback, stripped stereo, …). Always
    paired with a non-empty `status_note`.
  - `"human"` — **never set by the agent.** Reserved for downstream
    curator tools that mark a row as human-verified or human-completed.

- `status_note` is a free-text string. Empty (`""`) for `"agent"` rows;
  always non-empty for `"agent-needs-human"`, naming the specific issue
  and what a curator should do about it.

The `aggregate.py` normaliser enforces this:

- missing, non-string, or sentinel-string `status_note` is rewritten to
  `""`;
- missing or invalid `status` defaults to **`"agent-needs-human"`**, with
  an auto-populated `status_note` ("Status field was missing or invalid in
  the extraction output; row flagged for manual verification") when the
  agent didn't write its own. The reasoning: `"agent"` means the agent
  *affirmatively* signalled confidence — silently treating absence-of-
  signal as confidence would let weird rows pass without review. Anything
  the normaliser can't classify goes to the curator queue;
- `"human"` written by the agent becomes `"agent-needs-human"` with a
  flag (the agent's note is kept);
- an `"agent"` row with a non-empty `status_note` becomes
  `"agent-needs-human"` — the agent saw something worth saying;
- every problem the validator still finds, and every value a normaliser had
  to drop or coerce, is appended to `status_note` as one flag
  (`validation: …`, `field normalisation: …`, `catalytic normalisation: …`,
  `subunits normalisation: …`, `status normalisation: …`). Flags are
  deduplicated, so re-aggregating changes nothing.

The agent's `"agent-needs-human"` signal — when validly emitted — is
**always preserved**, even if the accompanying note ends up empty after
normalisation. A needs-review flag without explanation is still a
needs-review flag — the agent saw something it wasn't confident about,
and that signal is what gates the curator queue.

The benchmark scorer (`src/benchmark/evaluate.py`) ignores both fields —
they're metadata for the curation workflow, not for matching.

## Extracting arbitrary papers (no benchmark)

The pipeline runs on any DOI — no ground truth, registry entry, or local PDF
required (the agent fetches each paper via `paper_api.get_paper(DOI)`). A papers
list is just `[["doi"], ...]`:

```bash
echo '[["10.1038/s41586-020-2649-2"]]' > my_papers.json
python -m src.extract.batch my_papers.json --name my_run   # → data/processed/runs/<ts>_my_run/
python -m src.extract.run --doi 10.1038/s41586-020-2649-2  # single paper
```

Ad-hoc runs are **production mode**: the agent's shell keeps the network (it can
look things up on the web) but never sees the repository. See
`data/processed/README.md` for the full recipe, the run-folder layout and the
inspection recipes.

## Isolation of the extraction agent

Every attempt at a paper runs in its own folder outside the repo,
`$GARDEN_WORK_ROOT/<run>/<safe_doi>/attempt-<n>/{work,tmp,config}` (default
`~/.cache/garden-work`), as `claude -p --bare --restricted` with its own
`CLAUDE_CONFIG_DIR`, `TMPDIR` and `--session-id` (`src/extract/isolation.py`).
`--bare` offers only **Bash, Read and Edit** (no Write, WebFetch, WebSearch or
sub-agents); the agent writes its JSON array to `work/output.json` from the
shell.

- **File tools** reach only `work/`, the paper cache and the attempt's own session
  folder.
- **Bash** runs inside bubblewrap (`scripts/agent_sandbox.sh`, installed as
  `CLAUDE_CODE_SHELL_PREFIX`) with an allow-list root: the repo, the rest of
  `$HOME`, `/mnt/datasets`, other runs and the host `/tmp` do not exist inside;
  secrets are unset; shadow-library hosts (sci-hub, libgen, …) are blocked.
- **The shell keeps the network in both modes.** Production and benchmark runs
  are the same agent: same prompt, same tools, same internet.
- **Benchmark mode** (`--mode benchmark`, always passed by
  `scripts/run_benchmark.sh`) is production minus the benchmark's **answer key**
  (`--answer-key`, from the registry), and nothing else. The answer-key sources
  are defined in `src/extract/leak_signatures.py`: `uniprot` (uniprot_api and the
  Swiss-Prot BLAST `protein_sequence_to_ids`; uniprot.org, expasy.org), `rhea`
  (rhea_api; rhea-db.org) and `marts_db` (tpsdb.uochb.cas.cz and its Zenodo
  releases). swissprot50 removes `uniprot` + `rhea`; marts30 removes `marts_db`.
  Their MCP servers are withheld, their hosts resolve to nothing inside the
  sandbox, and the leak check recognises their data. Never add a benchmark-only
  tool or swap what a tool does: a source is only ever removed.
- **MCP servers** start through `scripts/mcp_launch.sh <server>` (the only command
  the sandbox wrapper passes through unsandboxed) and receive
  `GARDEN_WITHHELD_SERVERS`. A withheld server refuses every call; with
  `uniprot_api` withheld, `protein_sequence_to_ids` refuses too and `genbank_api`
  drops UniProt cross-references and refuses NCBI's copies of Swiss-Prot entries.
- **Preflight** (before any paper): every answer-key source is defined, a sandbox
  self-test passes (open hosts resolve, the answer key's hosts do not), and a
  Claude Code version not yet in `$GARDEN_WORK_ROOT/verified_cli.json` passes
  `scripts/isolation_spike.py` (a few Haiku calls, ≈ $0.20). Any failure stops the
  batch before spending.
- **Configuration gate** (`src/extract/preflight.py`, right after the preflight): the
  agent runs only in the set-up it was benchmarked in. Every deviation is listed —
  a server of `.mcp.json` that does not start, no MinerU VLM service or `mineru`
  CLI, no MolScribe service (or local CUDA GPU), no Java, a per-paper limit under
  3600 s, fewer than 2 repair rounds — and any of them refuses the run (exit 2, no
  run folder). `--allow-degraded` (`ALLOW_DEGRADED=1` for the launcher) runs anyway
  and records them in `run_config.json` (`configuration` per invocation,
  `configuration_complete` for the run). The model is a free parameter: recorded,
  never policed. `python -m src.extract.doctor` prints the same checks.
- **Leak check** (`src/extract/leakage.py`, after every attempt): scans the main
  and sub-agent sessions, persisted tool output and the sandbox log for evidence
  of access: answer-key data in any tool result (record formats, wherever it came
  from, including files the agent downloaded), requests for it on hosts that stay
  open (NCBI's copies of Swiss-Prot entries, EBI's UniProt API, MARTS-DB's Zenodo
  records), a file outside the agent's folders, an unsandboxed shell call; a
  mention is not a leak. Attempts to reach a blocked host are counted
  (`answer_key_host`). In benchmark mode a leaking attempt is discarded. If the
  agent reached the answer key itself, the paper gets one retry
  (`EXTRACT_ANSWER_KEY_RETRIES`) and then fails with error `answer_key`; if the
  isolation failed (a withheld server answered, an unsandboxed shell, …) the
  attempt is retried and 3 such failures stop the batch (exit 4).
  `python -m src.extract.leakage RUN_DIR` writes `leakage_report.json` (exit 0
  clean, 1 not, 2 checker error); the launcher scores only a clean run.

Environment knobs: `GARDEN_WORK_ROOT` (attempt folders), `GARDEN_KEEP_WORKDIRS=1`
(keep attempt folders after the transcript is saved), `GARDEN_CONDA_ROOT` (the
Python installation the sandbox may read; default: the conda root or prefix of
the interpreter running the pipeline), `GARDEN_PYTHON` (the interpreter
`scripts/mcp_launch.sh` starts the servers with; default: the pipeline's
interpreter), `GARDEN_OCR_PYTHON` (the OCR backends' interpreter; default: the
`ocsr-bench` conda env), `GARDEN_MINERU_BIN` (the mineru CLI; default: `mineru`
on PATH, else the `mineru-server` conda env's). The runner sets
`GARDEN_WITHHELD_SERVERS` and the `GARDEN_SANDBOX_*` variables itself.
`.mcp.json` spells the launcher and the paper folders relative to the
repository; the runner makes them absolute per attempt.
`python -m src.extract.doctor` checks a machine's set-up (packages, servers,
Java, the `claude` CLI, the sandbox, the key, the folders, the GPU services);
every line must be OK for the configuration gate to let a run start.

## Benchmark

**MARTS-DB** (https://tpsdb.uochb.cas.cz/?page=1) — a manually curated terpene synthase
database. Used as ground truth to validate agent extraction accuracy before scaling to
new papers and enzyme families.

Benchmark data lives in `data/benchmark/`. Only `reactions.csv` is used for benchmarking
(~4.5k rows). Key columns: Enzyme_marts_ID, Uniprot_ID, Genbank_ID, Enzyme_name,
Aminoacid_sequence, Species, Kingdom, Type, Class, Substrate_name, Substrate_smiles,
Product_name, Product_smiles, Publication_doi, Product_yield, Experiment_conditions.
The `mechanisms.csv` file is not needed for benchmarking.

### Fair evaluation protocol

**Never look at ground truth before or during extraction, and never add
benchmark-only behaviour** — a benchmark run is the production agent minus the
benchmark's answer key. The always-active rule
`.claude/rules/no-benchmark-shortcuts.md` spells this out; the isolation above
enforces it for the agent, and guard tests keep ground-truth accessions out of
the prompt and every server's tool text.

### Running the benchmark

Benchmarks are defined in a single registry — **`src/benchmark/benchmarks.py`**
(one entry per benchmark: paper list, ground-truth CSV, matching flags, and the
answer key removed at extraction to prevent GT leakage). Current benchmarks:
`marts30` (30 MARTS-DB terpene-synthase papers) and `swissprot50` (50
multi-family Swiss-Prot papers). Add a benchmark = add one entry there.

**swissprot50 ground truth is v2** (`gt_version="v2"`, 197 reactions, in
`data/benchmark/swissprot/release_2026-10-04_v2/` with its own `papers.json`):
v1 (`release_2026-06-02`, 199 reactions) carried a degenerate oleate→oleate
transport reaction and a duplicated propanal reaction. The benchmark keeps its
name; every results file records `gt_version`. Results scored against v1 are
never re-scored.

**One unified launcher** drives extraction + scoring for any benchmark (on a
GPU node; serves the model via vLLM, runs `batch.py` in benchmark mode with the
registry's answer key, gates on the leak summary, then scores the run's
own papers):

```bash
sbatch scripts/launch_molscribe_service.sh        # OCR service (once, on l40s)
sbatch scripts/launch_mineru_service.sh           # PDF→markdown VLM service (once, on l40s)
BENCHMARK=swissprot50 sbatch scripts/run_benchmark.sh
BENCHMARK=marts30     sbatch scripts/run_benchmark.sh
# smoke: LIMIT=5 or PAPERS=<list.json> ; resume: RESUME_RUN_DIR=<dir> ; other model: MODEL/VLLM_TP/TOOL_PARSER/…
# Anthropic API instead of vLLM (no GPU): PROVIDER=anthropic MODEL=claude-opus-5-5 bash scripts/run_benchmark.sh
```

It refuses to start (exit 2, no run folder) on any deviation from the
benchmarked configuration (`ALLOW_DEGRADED=1` runs anyway and records it), and
refuses to resume run folders made before the 2026-10 fixes. Exit status: the batch's, else the
leak summary's, else the scorer's; 0 only when all passed.

Or run the steps by hand (the scorer takes `--benchmark <name>`, which resolves
the GT CSV + matching flags from the registry):

```bash
python -m src.extract.batch data/processed/papers_30.json --parallel 4 --name marts30 --mode benchmark --answer-key marts_db
python -m src.extract.aggregate --run-dir data/processed/runs/<ts>_marts30   # automatic after every batch
python -m src.extract.leakage data/processed/runs/<ts>_marts30 --expect-mode benchmark --expect-answer-key marts_db
python -m src.benchmark.evaluate data/processed/runs/<ts>_marts30/extracted_reactions.json \
    --benchmark marts30 --papers data/processed/runs/<ts>_marts30/papers.json \
    --output-json data/processed/runs/<ts>_marts30/benchmark_results.json
```

**Batch exit codes:** 0 ok (failed papers are recorded in the metadata, not
fatal) · 2 a start-up guard, configuration or leak-checker error · 3 aggregation
failed · 4 leak breaker · 130 interrupted. **Scorer exit codes:** 0 ok (also for
an empty extraction) · 1 usage or unreadable input · 2 ground-truth error.

Matching is **direction-agnostic + (stereo-preserving) tautomer-canonical** for
all benchmarks by default (`--no-direction-agnostic` / `--no-tautomer` to opt
out); InChI-key, subset-containment, protonation-agnostic. Without `--papers`
the scorer scores every GT paper plus every DOI the extraction names, so the GT
CSV **must equal the benchmark's paper set**; for MARTS that's
`data/benchmark/marts_papers30/` (the full DB filtered to `papers_30`), not the
full 622-paper release. `--papers <run>/papers.json` restricts GT and rows to
the papers a run actually ran (the launcher always passes it). Negative-result rows
(`activity: "inactive"`) are never scored — the scorer, `reaction_diff` and the
HTML report drop them (shared predicate `src/utils/activity.py:is_negative`) and
`benchmark_results.json` reports their count per paper as `negative_rows`.

### Metrics (scorer v2, `src/benchmark/metrics.py`)

One module computes the numbers for the scorer, the HTML report,
`model_compare` and `perf_compare`, so they always agree.

- **Reaction recall** = GT reactions covered by some extracted row ÷ GT
  reactions. **Reaction precision** = extracted reaction rows that match a GT
  reaction ÷ *all* extracted reaction rows of the paper. Identical rows count
  once and negatives never count. A row for an enzyme absent from the GT (a
  mutant, a homologue, a wrong sequence) counts as unmatched, so low precision
  can simply mean the paper holds more data than the GT.
- **Enzyme precision** = distinct extracted enzymes that match any GT enzyme ÷
  distinct extracted enzymes. An extracted enzyme is its set of chains, each
  identified by its canonical sequence (or UniProt ID without a sequence); it
  matches when any chain equals a GT enzyme of the paper by UniProt ID or by
  sequence (identical, or one contained in the other, ≥ 20 residues). A complex
  counts once, as does an enzyme listed in many rows. **Enzyme recall** = GT
  enzymes found ÷ GT enzymes.
- **Substrates / products** compare the paper's pooled molecules; cofactors and
  small co-species (H₂O, H⁺, NAD(P)(H), ATP, CoA, Pi, PPi, …: the fixed table in
  `src/utils/cofactors.py`) are left out of both pools.
- **Over papers:** macro recall and macro F1 average **every GT paper** (a paper
  with no extraction scores 0); macro precision averages the papers with at
  least one extracted item and reports how many; micro scores pool the counts.
- **Results file** `benchmark_results.json` = `{meta, summary, papers}`. `meta`
  records the scorer version and git SHA, the benchmark and `gt_version`, the
  matching flags, and the paths and sha256 of the GT and the extraction. Read it
  with `metrics.load_results()`, which refuses files with impossible values. The
  comparison scripts refuse to mix old-format and v2 results, or different GT
  versions, unless `--allow-legacy` / `--allow-mixed-gt` is passed.

See `data/processed/README.md`
for the run-dir layout and comparison-across-runs recipes.

## Environment

- **Any Linux machine** with bubblewrap: `README.md` → Quick start.
  `scripts/setup_env.sh` installs the Python side into the active environment,
  `python -m src.extract.doctor` checks everything.
- **The authors' cluster** (Clara HPC, see `.claude/skills/clara-hpc-environment/`):
  conda at `~/miniconda3` (`source ~/miniconda3/bin/activate enzyme-mining`), shared
  datasets under `/mnt/datasets/bio/`, GPU services as Slurm jobs on the `l40s`
  partition. Nothing in the code depends on this layout — every path comes from the
  environment with a general default (the `GARDEN_*` knobs above).
- **Java**: required by `name_to_smiles` (OPSIN runs as a subprocess):
  `conda install -c conda-forge openjdk`.
- **Paper access**: `paper_api` downloads by DOI (publisher, PubMed Central); PDFs in
  `data/raw/pdfs/` are the fallback for papers it cannot fetch
- **LLM**: Claude Code (`claude -p --bare`) is the agent, driving either the Anthropic API
  (key from `$ANTHROPIC_API_KEY`, else read by the entry points from a chmod-600
  `~/.anthropic_api_key`; `ANTHROPIC_API_KEY_FILE` overrides the path) or a local
  vLLM-served model (`ANTHROPIC_BASE_URL`)

### MCP servers (bioinformatics/cheminformatics capabilities)

Nine servers from the sibling monorepo `bio-mcp-servers` (GitHub
`anton-bushuiev/bio-mcp-servers`, cloned next to this repository), wired in
through `.mcp.json` and each
started through `scripts/mcp_launch.sh <server> [<backend>]`: `paper_api`,
`uniprot_api`, `genbank_api`, `protein_sequence_to_ids`, `name_to_smiles`,
`catalytic_parameters`, `catalytic_residues_api`, `rhea_api`, `ocr_api`
(MolScribe by default). What each returns, how they are launched, withheld and
cached, the OCR backends and the checklist for adding a server are in the
**mcp-servers** skill (`.claude/skills/mcp-servers/SKILL.md`). Tool outputs
never contain `null`/NaN, and a result the agent should not trust blindly
carries a non-empty `warning`.

### Skills (on-demand, loaded when relevant)

- **mcp-servers** — The nine MCP servers: what they return, launch / withhold mechanics, OCR backends, caches, adding one
- **clara-hpc-environment** — Slurm GPU jobs, conda, dataset storage on Clara HPC
- **bioinformatics** — Protein structure file handling (biopandas, Bio.PDB)
- **molecule-visualization** — RDKit molecule / reaction SVGs for reports and diffs

### Rules (always active)

- **no-benchmark-shortcuts** — Ground truth never guides extraction; benchmark mode = production minus the answer key
- **visualization** — Publication-quality color palette and matplotlib plotting defaults
- **git** — Auto commit and push conventions
- **src-organization** — Code layout conventions for `src/` and `scripts/`

## Commands

```bash
# Activate the project environment (the one scripts/setup_env.sh was run in)
conda activate enzyme-mining
```

## Project Layout

```
src/
  extract/                # Extraction pipeline
    prompt.py             # Extraction prompt + output schema (26 fields) + REPAIR_PROMPT
    run.py                # One paper: attempts, repair rounds, leak check, per-paper file, metadata row
    batch.py              # Parallel batch over a paper list: guards, run_config, papers.json, aggregation
    isolation.py          # Per-attempt folders, claude argv/env, sandbox self-test, CLI version guard
    claude_cli.py         # Runs one `claude -p` attempt (process group, timeout, failure classes)
    transcripts.py        # Saves each attempt's session, sub-agents, tool results, envelope
    agent_output.py       # Reads the agent's output file (escape repair, row extraction)
    repair.py             # Repair rounds: resume the session with the validator's problem list
    validate.py           # Shared row validator (runner repair rounds + aggregator flags)
    leakage.py            # Leak check per attempt + run-level leakage_report.json
    leak_signatures.py    # Answer-key sources: servers, hosts, record formats, requests
    aggregate.py          # Per-paper files -> extracted_reactions.json (normalise, validate, flag)
    normalize_*.py, subunits.py  # Aggregate-time normalisers (common, fields, catalytic, chains)
    metadata.py           # extraction_metadata.json: locked appends, outcomes, accounting
    preflight.py          # GPU-service endpoint probes
  benchmark/              # Benchmarking against ground truth
    benchmarks.py         # Registry: paper list, GT CSV, gt_version, flags, answer key
    gt.py                 # Ground-truth loader (NA tokens, ';'-joined molecules, enzyme IDs)
    evaluate.py           # Scorer: InChI-key matching, subset-containment reactions
    metrics.py            # P/R/F1 per paper, macro/micro aggregation, results loader
    reaction_diff.py      # Per-reaction diff of a run vs GT (same binding as the scorer)
    build_swissprot.py    # Builds the Swiss-Prot benchmark releases
  utils/                  # Shared helpers (SMILES work lives in the name_to_smiles MCP server)
    doi.py                # canonical_doi(): the one DOI key used by runner, aggregator, scorer
    paper_file.py         # Per-paper extraction file {doi, attempt, session_id, rows}
    cofactors.py          # Fixed cofactor / co-species table (scorer + benchmark builder)
    balance.py            # Mass/charge balance (parity table in utils/data/)
    subunits.py, activity.py, atomic.py, provenance.py, tool_text.py
scripts/
  run_benchmark.sh        # Benchmark launcher (extraction -> leak gate -> scoring)
  agent_sandbox.sh        # bubblewrap wrapper for every agent shell command
  mcp_launch.sh           # The one MCP server launcher the sandbox lets through
  isolation_spike.py      # Live isolation check of a Claude Code version (≈ $0.20)
  launch_*_service.sh     # GPU services: MinerU (PDF -> markdown), MolScribe / DECIMER (OCR)
  vllm_*.py               # vLLM /v1/messages compatibility (schema patch, image-pruning proxy)
  analysis/               # Run comparison and HTML reports (model_compare, perf_compare, build_*_report)
  ocsr/                   # Chemistry-OCR benchmark (gold set, runner, viewer)
  presentations/          # Slide-deck and figure builders
data/
  benchmark/        # Ground-truth databases (MARTS-DB, Swiss-Prot releases)
  raw/pdfs/         # Downloaded paper PDFs
  processed/
    papers_*.json             # input lists for batch.py
    .smiles_cache.json        # legacy cache (gitignored, will retire)
    runs/<ts>[_<name>]/       # one directory per benchmark run (see README there)
  legacy/                     # old-schema artifacts, kept for diff/history
assets/
  figures/                    # README figures (SVG + PNG)
  reports/                    # benchmark reports, audits, surveys
  presentations/              # slide decks and their figures
```

## Conventions

- Python 3.11+
- Use `pip` for package management inside the `enzyme-mining` conda env
- Store benchmark data and results under `data/`