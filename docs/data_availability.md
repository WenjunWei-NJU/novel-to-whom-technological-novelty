# Data availability

## Observed study data

| Item | Verified study rule or evidence | Version status |
|---|---|---|
| PatentsView focal cohort | US utility patents filed January 1, 2000–December 31, 2024 and granted by December 30, 2025, as specified in the final manuscript. | Download package and provider release **VERSION NOT VERIFIED**. |
| Historical patent corpus | US utility patents granted January 6, 1976–December 30, 2025, including filings before 2000. Strictly prior records have application dates earlier than the focal date. | These are observed grant boundaries. They do not identify a provider download release. |
| CPC assignments | The study retrospectively uses the current grant-snapshot CPC assignments at subclass level. This supplies a common classification for current and historical patents. | Exact provider CPC snapshot **VERSION NOT VERIFIED**. At-application or at-issue classification is outside the certified input. |
| AIPat scores | The research input registry records 8,531,136 patent scores and SHA-256 `7a7889bb4f85e084b58c2750cf8045e0d23612fd78f5b535b6a522393b5f8a23`. The baseline uses scores ≥0.5, and missing scores remain unclassified. | Original prediction-file version and correspondence to a provider release **VERSION NOT VERIFIED**. |

The AIPat fingerprint identifies an archived transformed research input. Original source-file names, download timestamps and a checksum chain back to the provider release remain to be confirmed. Internal archive locations are excluded from this repository.

## Sources and terms

PatentsView supplies disambiguated patent, inventor, date and CPC records. The official data catalog lists CC BY 4.0 for the PatentSearch API. The applicable terms for the actual frozen bulk inputs and attribution of derived files still require verification, **LICENSE REVIEW REQUIRED**. [USPTO PatentsView](https://www.uspto.gov/ip-policy/economic-research/patentsview), [official API catalog](https://catalog.data.gov/dataset/patentsview-patentsearch-api-version-2-3-0).

AIPat provides patent-level AI scores. Cite Wu, T., Min, C., Ding, W. W., Wang, G., and Zhang, K. (2026), “Data and methods for identifying artificial intelligence-related patents.” The provider's current repository exposes prediction files with a 20260629 suffix, described as their USPTO source-data date. Their equivalence to the frozen input above is **VERSION NOT VERIFIED**. No explicit data-redistribution license was located in the reviewed top-level repository and README. **LICENSE REVIEW REQUIRED** applies to the study's derived distributions and display sample. [AIPat paper](https://doi.org/10.1016/j.respol.2026.105599), [provider repository](https://github.com/yusu2022/AIPat-Data-and-Methods-for-Identifying-Artificial-Intelligence-Related-Patents).

The online source and terms check was performed on November 17, 2026. Current public availability does not establish permission to redistribute the frozen study inputs.

## Hosting scope and external inputs

This private repository contains research code and configuration, methods documentation, and tests that generate artificial fixtures. It contains no real study data, numerical result files, or rendered figures.

We plan to share aggregate figure inputs and frozen table results where the applicable permissions allow. The project will not redistribute the original PatentsView or AIPat databases or the full microlevel analysis database. Third-party materials remain subject to their source terms.

The Figure 3 renderer defaults to the aggregate-only variant. With separate authorized inputs, it retains the weighted density, summary values, age profiles and adjusted estimates. It omits individual display dots, producing a visible difference from the submitted figure. Its underlying grouped distribution includes cells with one observation and requires separate review before hosting.

The 12,000 real derived inventor–patent display observations used for the submitted Figure 3 will remain nonpublic. They were excluded from local verification of the aggregate-only variant. See [Data and execution](../data/README.md) for the required files. The study input access mechanism remains unconfirmed.
