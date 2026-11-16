# Project 04 starter: Can a forecaster handle a new pattern of missing readings?

This package uses a **synthetic six-sensor benchmark**, not the Beijing dataset. It is a controlled starting point, with three fixed seeds and an included generator. Claims must be limited to this benchmark unless you add real data.

Run `pip install -r requirements.txt`, then `python starter.py --quick`. The standard run is `python starter.py --seed 940`; repeat seeds 941 and 942. A small NumPy GRU is included and requires no GPU or deep-learning framework. Its equations and gradients are exposed so students can inspect them. You may replace it with a framework implementation after matching the baseline settings.

## What the first run means

Use 24 past readings to predict sensor 0 six time steps after the last input. The training inputs have scattered artificial missing readings; development inputs have eight-step block outages. Naturally missing readings are recorded separately. The comparison trains two identical eight-unit GRUs: one receives time since each sensor was last observed; the other receives zeros in those same input positions. A last-observed-value forecast is the persistence baseline. No future value is used to fill a past reading. Mean and scale are computed from training observations only.

Missingness mechanisms remove exactly the same additional number of readings per sensor on a given chronological partition. This does **not** guarantee the same missingness exposure or observation age in each input window. That remaining difference is a research issue, not something the count check resolves. The data have a 3.5% independently generated natural-missing process, used consistently across conditions. Targets with natural missingness are excluded. Artificial masks only remove model inputs, never change target selection.

## Turn the exercise into research

Choose a claim about why a forecasting method succeeds or fails when the missingness pattern changes. Separate at least two candidate explanations, such as the age of the most recent reading versus the loss of several sensors together. Adding an age input alone reproduces a known idea; it is not sufficient. Design an age-matched comparison or another intervention that can distinguish the explanations. Matching total missing fraction alone is not enough.

Build and test one training change, reliability diagnostic, or sharply scoped statement of when a baseline can be trusted. Include persistence and the age-aware GRU, match training budgets, and compare the same examples across methods. Specify the meaning and unit of independent replication; overlapping forecast windows are not independent observations. Report per-seed results and uncertainty, not only a curve from one seed.

Freeze the claim, preprocessing, method, and settings before `python starter.py --final --seed 940`. The final test uses a later time period and value-dependent missingness: high observed values are more likely to be hidden. This jointly changes time period and mechanism. To attribute effects to the mechanism, also evaluate your frozen methods on scattered and block masks on that same test period. Never tune on final results.

## Files and deliverable

`data/synthetic_940.npz`, 941, and 942 contain chronological observed arrays (NaN denotes natural missingness), natural-missing indicators, and sensor-0 targets. Train/development/test lengths are 2,800/1,000/1,000. Targets are for loss/evaluation only, never input imputation. The same underlying series is used across masks. `generate_data.py` reproduces all files. Synthetic data are dedicated to the public domain under CC0-1.0. Starting reading: Che et al. (2018), *Recurrent Neural Networks for Multivariate Time Series with Missing Values* (GRU-D). This starter uses age features; it does not implement the full GRU-D decay mechanism.

An acceptable report establishes a testable claim beyond “blocks are worse than scattered gaps,” tests a competing explanation, evaluates a proposed method/diagnostic/limitation on the withheld condition, and documents where the conclusion stops applying. A negative result can satisfy these requirements; baseline graphs alone cannot.

## Setup check and Colab

Run `python check_data.py` to check data integrity and the relevant numerical identities. In Colab, upload this ZIP using the Files pane, unzip it into a folder with `!unzip -qo p5_starter.zip -d p5`, run `%cd p5`, then `!python starter.py --quick`. The package has been tested on CPU with NumPy 2.3.5; hosted Colab itself has not been tested. `VALIDATION.json` records the local checks and standard-run times. The small starter is intentionally not the full research workload.
