# Reproducibility map

Figure redraw and frozen-result table export are blocked from the checkout alone. Separately held and authorized inputs are required. The local technical checks below used external figure inputs and the frozen display JSON. Model refitting requires further external inputs as detailed in the [data contract](../data/README.md). Earlier saved-estimate checks are identified separately from this package's export tests.

## Figures

| Figure | Calculation and input | Status |
|---|---|---|
| 1 | [plot_figure1.py](../scripts/06_results_export/plot_figure1.py). Historical reference and descriptives. composition.csv, zeros.csv, novelty_correlations.csv. | PASS with external inputs, matched embedded PNG pixels |
| 2 | [plot_figure2.py](../scripts/06_results_export/plot_figure2.py). Team composition surface. composition_surface.csv, profile_support.csv, estimands.csv; A–C crop. | PASS with external inputs, matched embedded PNG pixels |
| 3 | [plot_figure3.py](../scripts/06_results_export/plot_figure3.py). Weighted grouped distributions, age profiles and stage estimates. | PASS with external aggregate inputs, matched the validated aggregate-only variant. The submitted PNG has individual display dots and visibly differs. |
| 4 | [plot_figure4.py](../scripts/06_results_export/plot_figure4.py). Peer-history coverage and opportunity. alluvial.csv, opportunities.csv, source_means.csv, estimands.csv. | PASS with external inputs, matched embedded PNG pixels |
| 5 | [plot_figure5.py](../scripts/06_results_export/plot_figure5.py). Threshold, window and period sensitivity. estimands.csv and plotted_estimates.csv cross-checked with saved formal estimates. | PASS with external inputs, matched embedded PNG pixels |

## Tables

| Table | Estimation or source entry | Verification |
|---|---|---|
| 1 | Team descriptive statistics and aggregate Figure 1 inputs. Descriptive calculation traced; frozen display export here. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| 2 | Three novelty outcomes and zero shares. Descriptive calculation traced; frozen display export here. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| 3 | Main model joint tests. run_main_models.py. | PASS with external frozen display data, plus prior saved-result check |
| 4 | Prespecified MAIN contrasts and cross-outcome test. run_main_models.py. | PASS with external frozen display data, plus prior saved-result check |
| 5 | Inventor full and stage models plus stage pairwise tests. run_personal_models.py and run_pairwise_stage.py. | PASS with external frozen display data, plus prior saved-result check |
| 6 | Peer-component and opportunity models. run_personal_models.py. | PASS with external frozen display data, plus prior saved-result check |
| S1 | Sample flow and model sample counts. build_core_base.py and model runners (partial flow). | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S2 | Descriptive aggregates and rank correlations. Descriptive aggregations; frozen export. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S3 | Planned Holm-family index. Planned definitions and family sizes in model runners. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S4 | Main team model coefficients. run_main_models.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S5 | Main model endpoint-support output. run_main_models.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S6 | Inventor model coefficients. run_personal_models.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S7 | Inventor full and stage contrasts and joint tests. run_personal_models.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S8 | Stage pairwise contrast output. run_pairwise_stage.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S9 | Peer-history coverage estimates. run_personal_models.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S10 | Weighted new-CPC-volume shares. Original descriptive summary; frozen export. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S11 | Opportunity-coverage descriptives and model. run_personal_models.py plus frozen descriptives. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S12 | Career-threshold results and endpoint support audit. run_sensitivity_batch.py and repair_threshold_support.py. | PASS with external frozen display data, plus prior saved-result check |
| S13 | Team model extensions. run_team_extensions.py. | PASS with external frozen display data, plus prior saved-result check |
| S14 | AI measurement alternatives. run_sensitivity_batch.py. | PASS with external frozen display data, plus prior saved-result check |
| S15 | Assignee M0/M1 contrasts. run_org_fe.py. | PASS with external frozen display data, plus prior saved-result check |
| S16 | Assignee match and sample report. run_org_fe.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S17 | Assignee endpoint report. run_org_fe.py. | PASS with external frozen display data. Full numeric source check NOT VERIFIED |
| S18 | Strict history and boundary sensitivity. run_history_completeness.py. | PASS with external frozen display data, plus prior saved-result check |
| S19 | Matched rolling-history sensitivity. run_sensitivity_batch.py. | PASS with external frozen display data, plus prior saved-result check |
| S20 | Period profiles, with pooled row in the same family. run_sensitivity_batch.py. | PASS with external frozen display data, plus prior saved-result check |
| S21 | Separate joint period inference. run_period_equality.py. | PASS with external frozen display data, plus prior saved-result check |
| S22 | Field contrasts and support. run_sensitivity_batch.py. | PASS with external frozen display data, plus prior saved-result check |
| S23 | Separate joint field inference. run_field_equality.py. | PASS with external frozen display data, plus prior saved-result check |

The preparation, estimation and sensitivity branches write numerical outputs and do not reproduce the typeset Word layout. Table links use the 23-table English SI numbering.
