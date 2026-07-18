# Israeli Election Bloc Prediction by Locality

A machine-learning project for predicting the distribution of votes among four political blocs in Israeli localities, using historical election results and demographic, socioeconomic, education, and locality-type features.

## Team Members

Ifat
Yuval
Roi

---

## Project Overview

Israeli election results vary substantially across localities. These differences are associated with demographic composition, socioeconomic characteristics, education, income, welfare indicators, and locality type.

The project develops a reproducible machine-learning workflow that predicts the vote-share distribution of four major political blocs for each locality:

- Right
- Center-Left
- Haredi
- Arab

Blocs are defined **coalitionally rather than ideologically**, reflecting the actual 2019–2022 coalition alignments: "Right" approximates the Netanyahu bloc without the Haredi parties, and "Center-Left" approximates the change bloc (and therefore includes Yisrael Beiteinu). A full sensitivity analysis with the alternative ideological tagging is included.

The project is designed as a proof of concept for locality-level election forecasting and political-demographic analysis.

---

## Business and Research Problem

Election forecasts are usually presented at the national level. National forecasts can hide strong geographic differences between cities, moshavim, kibbutzim, Arab localities, Druze-majority localities, and other settlement types.

The research question is:

> To what extent can demographic and socioeconomic characteristics explain and predict the distribution of votes among the four main political blocs at the locality level?

A locality-level model can support:

- Geographic political analysis
- Identification of localities with unusual voting patterns
- Scenario analysis
- Better understanding of the relationship between population characteristics and voting behavior
- Development of future election forecasting tools

The project does **not** attempt to predict individual voting behavior.

---

## Project Objectives

1. Combine election results from five election cycles with locality-level demographic data.
2. Map political parties into four consistent voting blocs, with an automated coverage audit.
3. Build normalized target variables whose shares sum to 100%.
4. Compare baseline and advanced regression models, including a neural-network baseline.
5. Prevent data leakage through chronological splitting and training-only preprocessing.
6. Select the model architecture on a dedicated validation election (Knesset 24) and report performance on a genuinely untouched test election (Knesset 25).
7. Analyze model errors by voting bloc and locality type.
8. Save a reproducible modeling pipeline for future forecasting.

---

## Target Variables

The target variables are normalized vote shares:

| Target | Meaning |
|---|---|
| `Right_pct` | Percentage of modeled votes assigned to the right-wing (Netanyahu-bloc, non-Haredi) parties |
| `Center_Left_pct` | Percentage of modeled votes assigned to the center-left (change-bloc) parties, including Yisrael Beiteinu |
| `Haredi_pct` | Percentage of modeled votes assigned to the Haredi bloc |
| `Arab_pct` | Percentage of modeled votes assigned to Arab parties |

The four target variables sum to 100% in each locality-election record.

Votes for parties that were not assigned to one of the four modeled blocs are retained in an audit variable called `Other`, but are excluded from the denominator used to calculate the four normalized targets. A dedicated stability analysis showed that `Other` has near-zero correlation across adjacent elections at the locality level and therefore cannot be modeled as a fifth bloc.

---

## Data Sources

The project uses aggregated public data at the locality level. All raw files are small and are included in `data/raw/` so the project runs end to end after cloning.

### 1. Election Results

Election results for the 21st through 25th Knesset elections were obtained from the official Israeli Central Elections Committee.

Official source:

- [Israeli Central Elections Committee](https://www.gov.il/en/departments/units/24-election-commitee/govil-landing-page)

Election cycles included:

| Election | Year |
|---|---:|
| Knesset 21 | 2019 |
| Knesset 22 | 2019 |
| Knesset 23 | 2020 |
| Knesset 24 | 2021 |
| Knesset 25 | 2022 |

### 2. Locality Characteristics

Locality-level explanatory variables were collected from public statistical sources: the Israel Central Bureau of Statistics (demographics, education, income and wages) and the National Insurance Institute (unemployment, benefits, and welfare indicators).

Official sources:

- [Israel Central Bureau of Statistics](https://www.cbs.gov.il/en/Pages/default.aspx)
- [National Insurance Institute of Israel](https://www.btl.gov.il/English%20Homepage/Pages/default.aspx)

The data includes variables from the following areas:

- Demographics and population structure
- Education
- Income, wages, and wage inequality (including a Gini index and a gender wage ratio)
- Unemployment
- Welfare and social-security indicators
- Population-group composition
- Locality classification

### 3. Locality-Type Classification

Localities were grouped into categories such as:

- Arab/Non-Jewish
- Cities
- Kibbutzim
- Moshavim
- Other

The Arab/Non-Jewish flag is also used as the routing variable of the segmented model. An additional binary variable, `type_Druze_Majority`, identifies Arab/Non-Jewish localities in which Druze residents represent at least 50% of the Arab population (computed as a locality-level median across years).

### Data Scope

The processed modeling dataset contains:

- **1,124 locality-election records**
- **226 unique localities** (localities large enough to have CBS profiles, covering ~82% of national valid votes)
- **5 election cycles**
- **4 target variables**
- **28 selected demographic features**
- Locality-type and engineered classification variables

## Methodology

### 1. Party-to-Bloc Mapping

Party-level election results were mapped into four consistent blocs using a party-keyed dictionary (each party listed once, with its bloc and every ballot letter it used across the five cycles), with an automatic collision check.

An automated audit cell quantifies unmapped ballot letters per election and fails loudly if any single unmapped letter exceeds 1% of valid votes. After adding independent runs of Otzma Yehudit and Gesher to the mapping, coverage exceeds 99% of valid votes in every cycle. Unmapped parties are aggregated into `Other` for coverage and quality-control purposes.

### 2. Data Integration

Election data was merged with annual locality-level features using:

- `locality_symbol`
- `year`

Because two elections took place in 2019, each election is also uniquely identified by `target_election`.

### 3. Chronological Model Selection and Testing

The data was split chronologically into three roles:

- **Selection training:** Knesset elections 21–23
- **Selection validation:** Knesset election 24 — used to compare all nine model architectures
- **Test:** Knesset election 25 — the winning architecture is retrained on elections 21–24 and evaluated **once** on this election

This design keeps the reported test metric free of model-selection bias and evaluates whether patterns learned from earlier elections generalize to a later election.

### 4. Missing-Value Treatment

Missing-value processing is layered and designed to prevent data leakage:

1. CBS missing-data tokens (`..`, `-`, etc.) are explicitly converted to NaN during numeric conversion.
2. Features with more than **30%** missing values in the selection-training set are removed.
3. **Selective LOCF:** a locality inherits a missing value from its own recent past (forward-fill only, at most two yearly observations back) — but only for features proven quasi-static on the training elections (year-over-year within-locality correlation of at least 0.9; 100 of 137 candidate features qualified). This step closed a real data-coverage break discovered in the 2022 CBS file.
4. Remaining missing values are imputed with a median `SimpleImputer` fitted on training data only, separately per model segment, and applied unchanged to validation and test data.

In Notebook 02, temporary median imputation is used only for training-based feature selection. The saved modeling dataset retains missing values so that each model pipeline can fit its own imputer correctly in Notebook 03.

### 5. Feature Selection

Candidate features were filtered using the selection-training set only.

The process included:

- Removal of features with more than 30% missing values in training
- Removal of fully missing and constant features
- Normalization of absolute welfare counts to rates per 1,000 residents
- Random Forest feature importance calculated separately for each target
- Union of the top-ranked features across all four targets (28 features selected)

Newly engineered income features — the 2019 wage Gini index (kept as a static locality attribute) and the male/female wage ratio — were selected by the Random Forest among the top predictors.

### 6. Compositional Modeling

The four target values form a composition because they must sum to 100%.

The project therefore evaluated:

- Independent regression predictions
- Post-hoc normalization
- Centered Log-Ratio transformation (`CLR`)

CLR converts the vote-share composition into an unconstrained log-ratio space (the inverse transformation is a softmax). Predictions are then transformed back into positive percentages that sum to 100% by construction.

### 7. Segmented Modeling

The strongest model uses separate CLR-XGBoost models for:

1. Arab/Non-Jewish localities
2. All other localities

Routing uses the external locality-type flag, which contains no missing values; imputation medians are learned separately per segment. The plain segmented model narrowly outperformed the variant with a Druze-majority indicator: the LOCF step restores the Arab-population-share feature, which makes the indicator largely redundant.

### 8. Sensitivity Analysis

The full pipeline was rerun with Yisrael Beiteinu tagged as Right (the ideological definition). The winning architecture family and the test-level accuracy are comparable under both definitions, supporting the robustness of the results; the coalitional definition is preferred for interpretability and for the stability of the model ranking.

---

## Models Evaluated

The following approaches were tested:

- Linear Regression
- Random Forest
- XGBoost using demographic features
- XGBoost using demographic and locality-type features
- XGBoost with post-hoc prediction normalization
- CLR-XGBoost
- A small neural-network baseline (sklearn MLP trained in CLR space, L2 + early stopping)
- Segmented CLR-XGBoost
- Segmented CLR-XGBoost with a Druze-majority indicator

The main evaluation metrics are:

- **MAE:** Mean Absolute Error, measured in percentage points
- **R²:** Proportion of variance explained by the model

---

## Main Results

Model-selection results on the Knesset 24 validation election (all nine architectures, trained on Knesset 21–23):

| Model | Selection MAE | Selection R² |
|---|---:|---:|
| **Segmented CLR** | **4.635** | **0.899** |
| Segmented CLR + Druze Indicator | 4.645 | 0.897 |
| XGBoost + Locality Types — Post-hoc Normalized | 4.900 | 0.904 |
| CLR-XGBoost — Demographic + Locality Types | 4.975 | 0.882 |
| XGBoost — Demographic + Locality Types | 5.115 | 0.888 |
| XGBoost — Demographic | 5.384 | 0.869 |
| Random Forest — Demographic | 5.468 | 0.867 |
| MLP (CLR) — Neural Baseline | 6.663 | 0.812 |
| Linear Regression — Demographic | 11.747 | 0.529 |

The complete comparison is generated by Notebook 03 and saved to:

- `reports/model_comparison.csv`
- `reports/model_comparison_by_target.csv`

### Selected Model and Final Test Result

The selected model is:

> **Segmented CLR** (retrained on Knesset 21–24, evaluated once on Knesset 25)

| Metric | Value |
|---|---:|
| Test MAE (Knesset 25) | **4.207** |
| Test R² | **0.916** |
| Vote-weighted test MAE | 2.723 |
| Test MAE — Right | 5.98 |
| Test MAE — Center-Left | 5.71 |
| Test MAE — Haredi | 2.16 |
| Test MAE — Arab | 2.98 |

Interpretation of the main metrics:

- An overall test MAE of approximately **4.2** means that the predicted bloc share differs from the actual share by about four percentage points on average.
- An overall R² of approximately **0.916** means that the model explains about 92% of the variance in the four bloc shares across the 2022 test records.

R² should not be interpreted as "92% prediction accuracy."

---

## Key Findings

1. Demographic variables alone provide useful but limited predictive power.
2. Locality-type variables materially improve model performance.
3. Treating the targets as compositional data improves predictions and guarantees valid vote-share outputs — but only in combination with segmentation.
4. Segmenting Arab/Non-Jewish localities from other localities produces the largest improvement.
5. The neural-network baseline underperformed the tree ensembles, consistent with the literature on small tabular datasets.
6. A real data-coverage break was discovered in the 2022 CBS file (the immigrants-since-1990 share is missing for 72% of Arab localities); the selective LOCF step closed it and made the pipeline robust to this data drift.
7. The evaluation design keeps the test election entirely outside the selection loop, because validation differences between top architectures can be smaller than the noise level — the reported test metric is therefore free of selection bias.
8. Distinguishing between Right and Center-Left remains more difficult than predicting strongly concentrated Haredi or Arab voting patterns.


## Reproducibility

The project follows the following reproducibility principles:

- Fixed `random_state` (99)
- Chronological selection / validation / test split
- Training-only imputation, filtering, feature selection, and LOCF eligibility
- Saved feature lists
- Saved model artifacts
- Explicit notebook execution order
- Repository-relative paths only
- Package versions stored in `requirements.txt`

---

## Error Analysis

The evaluation notebook includes:

- Actual-versus-predicted plots
- MAE and R² by target
- Error analysis by locality type (largest errors: Druze-majority localities and moshavim; cities: MAE 2.80)
- Top 10 localities with the largest errors
- Overprediction and underprediction analysis
- Vote-weighted MAE

The detailed results are saved under `reports/`.

---

## Limitations

1. Coverage is limited to the 226 localities with CBS profiles (~82% of national valid votes); small moshavim and kibbutzim are underrepresented, and the largest errors concentrate there.
2. Annual demographic data cannot fully represent changes between the two elections held in 2019, and those two elections share identical feature rows.
3. The model does not include campaign events, candidate changes, security events, turnout shocks, party mergers, or newly formed parties.
4. The relationship between demographic variables and voting patterns may change over time, and CBS reporting coverage itself drifts between years (as observed in 2022).
5. Some locality groups contain relatively few observations.
6. The model predicts locality-level bloc shares for known localities and was not evaluated on unseen localities; it does not predict national seat allocation.
7. Bloc definitions are coalitional by design; the ideological alternative is provided as a sensitivity analysis.
8. The current project is a proof of concept and is not yet a production election-forecasting system.

---

## Conclusions

The project demonstrates that locality-level demographic and socioeconomic information can explain a large part of the variation in historical voting patterns.

The strongest performance was achieved by combining:

- XGBoost
- CLR compositional modeling
- Segmentation by Arab/Non-Jewish locality type
- Selective locality-level LOCF imputation

The final result is a genuine held-out test MAE of approximately **4.2 percentage points** (2.7 when weighted by votes) and an R² of approximately **0.916** on the 2022 election.

---

## Repository Structure

```
election-bloc-prediction/
├─ README.md                  # this file
├─ requirements.txt           # pinned package versions
├─ .gitignore
├─ data/
│  ├─ raw/                    # source files (elections, CBS, NII, locality types)
│  ├─ processed/              # generated datasets (recreated by the notebooks)
│  └─ data_dictionary.md      # description of every table and column
├─ notebooks/
│  ├─ 01_data_mapping_and_merging.ipynb
│  ├─ 02_eda_and_preprocessing.ipynb
│  ├─ 03_modeling.ipynb
│  └─ 04_model_evaluation.ipynb
├─ models/                    # saved final model bundle (joblib)
├─ reports/
│  ├─ figures/                # saved charts
│  ├─ model_results.md        # concise performance summary
│  └─ *.csv / *.json          # comparison tables and summaries
├─ src/                       # placeholder for reusable functions
└─ presentation/              # slide deck
```

## How to Run

1. Clone the repository.
2. Create an environment with Python 3.12 and install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Launch Jupyter (`jupyter lab`) and run the notebooks **in order** from the `notebooks/` folder:
   `01 → 02 → 03 → 04`, each from start to finish.

Notes:

- All paths are repository-relative; no configuration is needed.
- The `data/processed/` outputs are included, so notebooks 02–04 can also be run without rerunning notebook 01.
- A full pipeline run takes a few minutes; notebook 03 (nine model fits) is the longest step.

---

## Next Steps

1. Retrain the selected model structure using all available elections, including 2022.
2. Prepare a single up-to-date demographic record for each locality (the LOCF step already handles partial CBS coverage).
3. Generate one future prediction per locality and evaluate the next election as a truly unseen test set.
4. Add a previous-election baseline for comparison.
5. Add temporal and political-event features, and evaluate leave-one-election-out validation.
6. Move reusable functions from notebooks into `src/`.
7. Add the final slide deck to `presentation/`.
