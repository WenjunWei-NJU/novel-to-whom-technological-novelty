# Verification report

| Check | Result |
|---|---|
| Checkout-only figure redraw and table export | BLOCKED. Study inputs and result files are excluded pending hosting clearance. Both commands report missing input clearly. |
| Figures 1, 2, 4 and 5 | PASS with separately held inputs. Redrawn PNGs match the submitted PNGs pixel for pixel. |
| Figure 3 aggregate-only variant | PASS with separately held inputs. Redrawn PNG matches the previously validated aggregate-only variant. The submitted figure visibly differs because it includes display dots. |
| Frozen table export | PASS with separately held input. All 29 CSVs and the notes index match the audited frozen exports. |
| Lightweight tests | PASS in an isolated copy with the previously checked local Python environment. Twenty tests passed. Production peer-history SQL completed 24,956 comparisons without a failure. |
| Missing-input behavior | PASS. The figure, table and analysis commands identify missing files before creating output. |
| Fresh environment installation | NOT VERIFIED. The test used the previously isolated package environment. |
| Formal full-sample refits and raw-data reconstruction | NOT VERIFIED in this package. |
| Source result comparisons | The previous audit checked 829 saved-result fields. This package was checked at the frozen display export level. |

The manuscript, supplement and original study artifacts served as read-only references. The hosted candidate excludes the real Figure 3 display sample, all figure and table inputs, model outputs and rendered artifacts pending clearance.
