# Novel to Whom? Historical Reference and the Measurement of Technological Novelty

Replication materials for the Journal of Informetrics submission by Wenjun Wei, Alex Jie Yang, and Sanhong Deng. The study compares career-stage composition in AI-related and non-AI patents using inventor, team, and prior-corpus historical references. The main analytical samples include 2,670,332 patents and 12,908,659 inventor–patent records.

## Historical references and outcomes

- **Mean personal novelty** averages the share of the focal CPC subclasses absent from each inventor's own earlier patent record over eligible team members.
- **Team novelty** measures the share of those CPC subclasses absent from the union of the team's earlier patent records.
- **Corpus-first pairs** measures the share of unordered CPC subclass pairs absent from patents filed strictly earlier, on patents with at least two subclasses.
- **Peer-history coverage** partitions personally new subclasses by whether early and other peers had previously recorded them. **Opportunity coverage (U)** measures the use of the early-peer-exclusive opportunity set when that set is nonempty.

Date order is strict at the application-date level. See the [variable dictionary](docs/variable_dictionary.md) for the career-stage definitions, FIRST cases, and denominators. The [data preparation commands](data/README.md#prepared-data-build) start with prepared inputs. They assemble team composition, analytical records, and rolling histories before estimation.

## Private review package

This private repository contains research code and configuration, methods documentation, and tests that generate artificial fixtures. It contains no real study data, numerical result files, or rendered figures. An open-source license has yet to be adopted.

| Level | Scope | Status in this package |
|---|---|---|
| A | Redraw Figures 1–5 | BLOCKED with checkout alone. The drawing code was tested locally with separately held inputs. Figures 1, 2, 4 and 5 matched the submitted PNGs. Figure 3 defaults to a visibly different aggregate-only rendering without individual display dots. |
| B | Export 29 frozen table displays | BLOCKED with checkout alone. The exporter was tested locally against the separately held frozen display JSON. |
| C | Re-estimate the formal models | NOT VERIFIED at full sample. The required external inputs include the study's frozen `core.duckdb`; see the [input contract](data/README.md). |
| D | Rebuild the formal analysis from original data | NOT VERIFIED. Provider version equivalence and the full upstream history and pair reconstruction remain unverified. |

## Run lightweight tests

```sh
conda env create -f environment.yml
conda activate novel-to-whom
python -m unittest discover -s tests -v
python tests/check_production_peer_history.py
```

If you separately hold the authorized figure inputs and frozen table JSON, point the scripts to those locations. The figure folder must contain `figure1` through `figure5` as listed in [the figure configuration](config/figures.json). The table exporter accepts the authored frozen display JSON for Tables 1–6 and S1–S23.

```sh
python scripts/06_results_export/redraw_figures.py --data-dir "$FIGURE_INPUT_DIR" --output output/figures 1 2 3 4 5
python scripts/06_results_export/export_tables.py --input "$FROZEN_TABLE_JSON" --output output/tables
```

Set `FIGURE_INPUT_DIR` and `FROZEN_TABLE_JSON` to your separate local input paths before running these commands. The Figure 3 entry point uses the weighted aggregate distribution, age profiles and estimates. It omits the individual display dots. The table command exports frozen displays and notes. Re-estimation requires the formal inputs described below.

## Estimate the models

After obtaining the study's frozen `core.duckdb` separately, place it at `$DATA/core.duckdb`, where `DATA` is the directory you select. The model scripts read the database and write separate results directories.

```sh
python scripts/04_main_analysis/run_main_models.py \
  --database "$DATA/core.duckdb" --output output/main

for kind in full stage source opportunity; do
  python scripts/04_main_analysis/run_personal_models.py "$kind" \
    --database "$DATA/core.duckdb" --output "output/personal/$kind"
done

python scripts/05_sensitivity_analysis/run_sensitivity_batch.py \
  --database "$DATA/core.duckdb" --output output/sensitivity
python scripts/05_sensitivity_analysis/repair_threshold_support.py \
  --database "$DATA/core.duckdb" --results output/sensitivity
```

The support audit updates the designated sensitivity output directory. Run it on a newly generated copy. Other sensitivity branches and the build from prepared inputs are documented in [Data and execution](data/README.md). The preflight and raw-event builder provide a separate method-validation route using small fixtures or supplied patent events. The raw-event route has not been certified to reconstruct all prepared inputs used in the submission.

## Repository layout

| Directory | Contents |
|---|---|
| `scripts/01_data_preparation` | Team roster aggregation |
| `scripts/02_inventor_histories` | Historical-window extension and separate event-based builder |
| `scripts/03_novelty_indicators` | Formal analytical-table assembly and the production SQL functions for member, team and peer history measures |
| `scripts/04_main_analysis` | Main team and inventor-level estimation |
| `scripts/05_sensitivity_analysis` | Sensitivity batches, threshold support, assignee models, stage pairwise and period and field joint inference |
| `scripts/06_results_export` | Figure programs, redraw entry point and frozen table exporter |
| `scripts/_shared` | Estimation, inference, input and testable history methods shared by the entry points |
| `config`, `data`, `tests`, `docs` | Configurations, input contracts, tests and documentation |

## Data, resources and citation

Sources are [USPTO PatentsView](https://www.uspto.gov/ip-policy/economic-research/patentsview) and [AIPat](https://doi.org/10.1016/j.respol.2026.105599). Data packaging, access and version limitations are in [Data availability](docs/data_availability.md). Plot reproduction and small tests fit a standard laptop. Formal refits need the analytical database, ample scratch space, and enough memory to hold large analysis arrays. The team sample is 2.67 million records and the inventor models use roughly 13 million records. The archived DuckDB input is about 1.5 GB. These are scale indicators, since no clean-machine runtime benchmark was run here.

See the [result map](docs/reproducibility_map.md), [variable dictionary](docs/variable_dictionary.md), and [verification report](docs/verification_report.md) for the evidence and limits. Cite the associated manuscript using [CITATION.cff](CITATION.cff).

We plan to share aggregate figure inputs and frozen table results where permissions allow. The project will not redistribute the original PatentsView or AIPat databases, the full microlevel analysis database, or the 12,000 real inventor–patent display observations used for the submitted Figure 3.
