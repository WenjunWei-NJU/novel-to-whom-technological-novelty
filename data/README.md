# Data and execution

This checkout contains input documentation. Obtain authorized inputs separately and pass their paths to the commands. The figure folder must contain only the required files listed for each figure in [config/figures.json](../config/figures.json). Figure 3 requires aggregate distribution, age-profile and estimate files. The table exporter expects a separately held frozen display JSON containing the 29 submitted table displays, including notes and merged-cell metadata.

## Frozen analytical input

The external study file `core.duckdb` is excluded from the repository. Its access mechanism awaits author confirmation. It contains `teams` and `personal` at patent and inventor–patent grain. The model preparation expects the rolling-history and age-distance columns for sensitivity checks. Supply the file with `--database`. Exact data comparable to the formal runs must match the study input version.

## Prepared data build

Set `DATA` to the directory containing the study's prepared datasets. The filenames below are an example. CLI inputs may be any matching user-supplied files.

| Input | Expected fields |
|---|---|
| `roster/part-*.parquet` | `patent_id, history_blob, current_blob, history_state, personal_Y, career_age_years, prior_patent_count, prior_cpc_count, prior_cpc_gap`. All records for a patent are in the same shard. CPC sets use big-endian bit masks. |
| `team_history.parquet` | `patent_id, filing_date, grant_date, application_year, ai_aipat50, aipat_ai_score, n_newcomer, n_veteran, n_unknown, first_observed_count, ibi_n5v16, team_existing_Y, n_full`. |
| `primary_cpc.parquet` | `patent_id, cpc_subclass`. |
| `pair_history/*.parquet` | `patent_id, K, new_count, new_pairs_exposed_to_undated`. |
| `personal_candidates/*.parquet` | `patent_id, inventor_id, application_year, n_full, E_early_peer_share, veteran_peer_share, first_peer_share, career_age_years, prior_patent_count, prior_cpc_count, personal_Y, career_stage, A_share, B_share, C_share, D_share, early_only_opportunity_n, early_only_utilization, P1_basic_variable_candidate, source_basic_variable_candidate, utilization_eligible`. |
| `history_events.duckdb` | `events` table with `patent_id, inventor_id, filing_date, shard`. Shards are 0–15. |
| `cpc_dictionary.csv` | `cpc_subclass, code_index`, original 680-bit encoding. |
| `current_cpc_sets.parquet` | `patent_id, current_cpc_set` as a list of CPC subclasses. |

```sh
python scripts/01_data_preparation/build_team_boundaries.py \
  --roster-dir "$DATA/roster" --output output/team-boundaries
python scripts/03_novelty_indicators/build_core_base.py \
  --team-boundaries 'output/team-boundaries/team_boundaries_part-*.parquet' \
  --team-history "$DATA/team_history.parquet" --primary-cpc "$DATA/primary_cpc.parquet" \
  --pair-history "$DATA/pair_history/*.parquet" \
  --personal-candidates "$DATA/personal_candidates/*.parquet" --output output/core
python scripts/02_inventor_histories/extend_history_base.py \
  --database output/core/core.duckdb --history-database "$DATA/history_events.duckdb" \
  --cpc-dictionary "$DATA/cpc_dictionary.csv" --current-cpc-sets "$DATA/current_cpc_sets.parquet" \
  --output output/with-history
```

The rolling-history step attaches window variables after the core tables exist. Use `output/with-history/core.duckdb` for refitting. The separate `build_from_patent_events.py` uses [a configuration template](../config/patent_events.example.json) for raw patent dates, inventor links, CPC subclasses and AI scores. It implements methods checked on artificial data. Its output is not certified as interchangeable with the formal frozen database.

## Additional sensitivity commands

Run these after the main-analysis and sensitivity commands in [README](../README.md). In the examples, `DATA` points to the study's analytical inputs plus any branch-specific data.

```sh
python scripts/05_sensitivity_analysis/run_team_extensions.py \
  --database "$DATA/core.duckdb" --output output/extensions
python scripts/05_sensitivity_analysis/run_history_completeness.py \
  --database "$DATA/core.duckdb" --patent-dates "$DATA/patent_dates.parquet" \
  --patent-inventors "$DATA/patent_inventor.parquet" --output output/completeness
python scripts/05_sensitivity_analysis/run_org_fe.py \
  --database "$DATA/core.duckdb" --main-results output/main \
  --anchor "$DATA/assignee_anchor.parquet" --qualification "$DATA/assignee_qualification.parquet" \
  --output output/assignee
for group in period field; do
  python "scripts/05_sensitivity_analysis/run_${group}_equality.py" \
    --database "$DATA/core.duckdb" --sensitivity-results output/sensitivity \
    --output "output/${group}-equality"
done
python scripts/05_sensitivity_analysis/run_pairwise_stage.py \
  --stage-results output/personal/stage --output output/pairwise
```

The completeness comparison needs patent grant dates plus a patent–inventor roster. The assignee comparison also needs validated organizational anchor and qualification tables. The anchor fields include `patent_id, anchor_assignee_id, anchor_is_organization, anchor_missing, mapping_failure_n, anchor_selection_rule, multiple_organization_assignees`. The qualification table supplies `patent_id, anchor_assignee_id, reliable_organization`. The separate Holm families and plotted outputs are described in [the result map](../docs/reproducibility_map.md).

## Required inputs by executable entry

All paths below are user-selected files or directories. The database and prepared study inputs are excluded from this repository. Access remains subject to author confirmation and applicable provider terms.

| Entry | Required inputs |
|---|---|
| `build_team_boundaries.py` | `--roster-dir` with `part-*.parquet`. |
| `build_core_base.py` | `--team-boundaries --team-history --primary-cpc --pair-history --personal-candidates`, matching the schemas above. |
| `extend_history_base.py` | `--database --history-database --cpc-dictionary --current-cpc-sets`. |
| `build_from_patent_events.py` | `--config` and all required raw event inputs in that configuration. Optional inputs are schema-checked when supplied. This route remains separate from the certified prepared database. |
| `run_main_models.py` | `--database` and `--config` (bundled analysis defaults). |
| `run_personal_models.py` | `kind`, `--database` and `--config`. Historical `source` and `opportunity` interface names mean peer-history coverage and opportunity coverage (U). |
| `run_team_extensions.py`, `run_sensitivity_batch.py` | `--database` with required window and team-age variables, and `--config`. |
| `run_history_completeness.py` | `--database --patent-dates --patent-inventors` and `--config`. |
| `repair_threshold_support.py` | `--database --results` with `analysis_settings.json`, `tables/estimands.csv`, and `tables/profile_support.csv` from the sensitivity batch. |
| `run_org_fe.py` | `--database --anchor --qualification --main-results`, including main settings, estimands and composition support. |
| `run_period_equality.py`, `run_field_equality.py` | `--database --sensitivity-results`, including sensitivity settings and estimands. |
| `run_pairwise_stage.py` | `--stage-results` with `models/personal_stage.json` and `tables/estimands.csv`. |
| `redraw_figures.py` | Selected figure numbers, `--output` and separately held inputs declared in `config/figures.json`. Figure 3 uses aggregate inputs. |
| `export_tables.py` | `--output` and `--input` pointing to the separately held frozen table JSON. |

Level A redraws figures. Level B exports frozen table displays. Level C refits formal models with external study inputs. Level D rebuilds those inputs from original provider records. Verification of each level is summarized in the [README](../README.md#private-review-package).
