# Variable dictionary

| Variable or quantity | Definition |
|---|---|
| Current CPC set | Deduplicated four-character CPC subclasses assigned to a patent |
| Strictly prior records | Patents filed strictly before the focal filing date. Same-day filings do not enter one another's histories. |
| Observed career age | Elapsed days from the inventor's first observed filing to the focal filing divided by 365.25 |
| Early, mid, senior | Early at most 5 years, mid above 5 and below 16, senior at least 16 |
| FIRST | Verified first observed filing, zero observed career age. In the team roster and early share, with empty prior history. Excluded from the eligible-person average. Unresolved histories are handled separately. |
| `y`, personal novelty | Current subclasses absent from the eligible inventor's prior history, divided by all current subclasses |
| `relative_y`, mean personal novelty | Mean of `y` across eligible inventors on a patent |
| `team_y`, team novelty | Current subclasses absent from the union of all team histories, divided by the same number of current subclasses |
| `pair_y`, corpus-first pairs | Fraction of unordered current CPC pairs absent from all strictly earlier observed filings. Undefined if current CPC count is below two. Same-day first observations retain their status. |
| `early_peers`, early-peer share | Share of other team members in the early stage, excluding the focal inventor |
| A, B, C, D, peer-history coverage | Components of `y`. The personally new current classes occur in only early-peer history, only other-peer history, both, or neither. Each component uses the current CPC count and the four sum to `y`. |
| `opportunity_n` | Size of the early peers' union after removing the focal history and other peers' union |
| `utilization`, opportunity coverage (U) | Current classes in the opportunity set divided by its size. Undefined when the set is empty and zero when the nonempty set has no current overlap. |
| `ai`, AI-related and non-AI | AIPat score at least 0.5 under the baseline. Missing scores remain unclassified. Reflects patent technology content. |
| Prespecified five-member composition contrast | AI minus non-AI difference in predicted change for five-member teams, FIRST fixed at zero, stage counts from (1,3,1) to (2,1,2). The direct personal-minus-team test uses cross-outcome covariance. |
| Weights and covariance | Team models use equal patent weights and primary-CPC clusters. Inventor models use initial patent-normalized weights and inventor–patent or inventor–primary-CPC two-way clustering. Holm-adjusted tests retain their specified families. |

## Historical interface names

`utilization`, `early_only_utilization`, `utilization_weight` and `opportunity_utilization` are retained field, weight and model identifiers for **opportunity coverage (U)**. The `source` command and `SOURCE_COMPOSITION` family refer to the A–D **peer-history coverage** analysis. `E` and `early_peers` encode **early-peer share**. Identifiers and Holm-family keys remain the interfaces used by the formal programs.

Frozen result files and figure artwork retain their submitted labels. Abbreviations such as Personal, Team and Corpus pairs refer to mean personal novelty, team novelty and corpus-first pairs in team-level displays.
