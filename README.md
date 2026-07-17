# Israeli Election Bloc Prediction by Locality

A machine-learning project for predicting the distribution of votes among four political blocs in Israeli localities, using historical election results and demographic, socioeconomic, education, and locality-type features.

## Team Members

| Team member | Main responsibility |
|---|---|
| Ifat  | |
| Yuval | |
| Roi | |

---

## Project Overview

Israeli election results vary substantially across localities. These differences are associated with demographic composition, socioeconomic characteristics, education, income, welfare indicators, and locality type.

The project develops a reproducible machine-learning workflow that predicts the vote-share distribution of four major political blocs for each locality:

- Right
- Center-Left
- Haredi
- Arab

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
2. Map political parties into four consistent voting blocs.
3. Build normalized target variables whose shares sum to 100%.
4. Compare baseline and advanced regression models.
5. Prevent data leakage through chronological splitting and training-only preprocessing.
6. Evaluate performance on the 2022 election.
7. Analyze model errors by voting bloc and locality type.
8. Save a reproducible modeling pipeline for future forecasting.

---

## Target Variables

The target variables are normalized vote shares:

| Target | Meaning |
|---|---|
| `Right_pct` | Percentage of modeled votes assigned to the right-wing bloc |
| `Center_Left_pct` | Percentage of modeled votes assigned to the center-left bloc |
| `Haredi_pct` | Percentage of modeled votes assigned to the Haredi bloc |
| `Arab_pct` | Percentage of modeled votes assigned to Arab parties |

The four target variables sum to 100% in each locality-election record.

Votes for parties that were not assigned to one of the four modeled blocs are retained in an audit variable called `Other`, but are excluded from the denominator used to calculate the four normalized targets.

---

## Data Sources

The project uses aggregated public data at the locality level.

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

Locality-level explanatory variables were collected from public statistical sources, primarily the Israel Central Bureau of Statistics.

Official source:

- [Israel Central Bureau of Statistics](https://www.cbs.gov.il/en/Pages/default.aspx)

The data includes variables from the following areas:

- Demographics and population structure
- Education
- Income and wages
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

An additional binary variable, `type_Druze_Majority`, identifies Arab/Non-Jewish localities in which Druze residents represent at least 50% of the Arab population.

### Data Scope

The processed modeling dataset contains:

- **1,124 locality-election records**
- **226 unique localities**
- **5 election cycles**
- **4 target variables**
- **25 selected demographic features**
- Locality-type and engineered classification variables

## Methodology

### 1. Party-to-Bloc Mapping

Party-level election results were mapped into four consistent blocs:

- Right
- Center-Left
- Haredi
- Arab

Unmapped parties were aggregated into `Other` for coverage and quality-control purposes.

### 2. Data Integration

Election data was merged with annual locality-level features using:

- `locality_symbol`
- `year`

Because two elections took place in 2019, each election is also uniquely identified by `target_election`.

### 3. Chronological Validation

The data was split chronologically:

- **Training:** Knesset elections 21-24
- **Validation:** Knesset election 25, held in 2022

This design evaluates whether patterns learned from earlier elections can generalize to a later election.

### 4. Missing-Value Treatment

Missing-value processing was designed to prevent data leakage.

- The dataset is split before model imputation.
- The `SimpleImputer` is fitted only on training data.
- The learned training medians are then applied unchanged to the validation data.
- The imputer is never fitted on the complete dataset before the split.

In Notebook 02, temporary median imputation is used only for training-based feature selection. The saved modeling dataset retains missing values so that each model pipeline can fit its own imputer correctly in Notebook 03.

### 5. Feature Selection

Candidate features were filtered using the training set only.

The process included:

- Removal of features with more than 50% missing values in training
- Removal of fully missing features
- Removal of constant features
- Random Forest feature importance calculated separately for each target
- Union of the top-ranked features across all four targets

### 6. Compositional Modeling

The four target values form a composition because they must sum to 100%.

The project therefore evaluated:

- Independent regression predictions
- Post-hoc normalization
- Centered Log-Ratio transformation (`CLR`)

CLR converts the vote-share composition into an unconstrained log-ratio space. Predictions are then transformed back into positive percentages that sum to 100%.

### 7. Segmented Modeling

The strongest model uses separate CLR-XGBoost models for:

1. Arab/Non-Jewish localities
2. All other localities

The Arab/Non-Jewish model also includes `type_Druze_Majority`.

---

## Models Evaluated

The following approaches were tested:

- Linear Regression
- Random Forest
- XGBoost using demographic features
- XGBoost using demographic and locality-type features
- XGBoost with post-hoc prediction normalization
- CLR-XGBoost
- Segmented CLR-XGBoost
- Segmented CLR-XGBoost with a Druze-majority indicator

The main evaluation metrics are:

- **MAE:** Mean Absolute Error, measured in percentage points
- **R²:** Proportion of variance explained by the model

---

## Main Results

Selected validation results:

| Model | Overall MAE | Overall R² |
|---|---:|---:|
| XGBoost — Demographic | 11.459 | 0.516 |
| XGBoost — Demographic + Locality Types | 8.238 | 0.740 |
| XGBoost + Locality Types — Post-hoc Normalized | 8.183 | 0.768 |
| Segmented CLR | 4.133 | 0.929 |
| **Segmented CLR + Druze Indicator** | **3.992** | **0.931** |

The complete comparison is generated by Notebook 03 and saved to:

- `reports/model_comparison.csv`
- `reports/model_comparison_by_target.csv`

### Selected Model

The selected model is:

> **Segmented CLR + Druze Indicator**

Interpretation of the main metrics:

- An overall MAE of approximately **3.99** means that the predicted bloc share differs from the actual share by about four percentage points on average.
- An overall R² of approximately **0.931** means that the model explains about 93.1% of the variance in the four bloc shares across the 2022 validation records.

R² should not be interpreted as “93.1% prediction accuracy.”

---

## Key Findings

1. Demographic variables alone provide useful but limited predictive power.
2. Locality-type variables materially improve model performance.
3. Treating the targets as compositional data improves predictions and guarantees valid vote-share outputs.
4. Segmenting Arab/Non-Jewish localities from other localities produces the largest improvement.
5. The Druze-majority indicator adds a small additional improvement.
6. Most remaining errors are concentrated in localities with unusual political patterns.
7. Distinguishing between Right and Center-Left remains more difficult than predicting strongly concentrated Haredi or Arab voting patterns.


## Reproducibility

The project follows the following reproducibility principles:

- Fixed `random_state`
- Chronological train-validation split
- Training-only imputation
- Training-only feature filtering and selection
- Saved feature lists
- Saved model artifacts
- Explicit notebook execution order
- Documented source folders and generated outputs
- Package versions stored in `requirements.txt`

Before submission, verify that all notebooks run from start to finish in a clean runtime.

---

## Error Analysis

The evaluation notebook includes:

- Actual-versus-predicted plots
- MAE and R² by target
- Error analysis by locality type
- Error analysis for Druze-majority localities
- Top 10 localities with the largest errors
- Overprediction and underprediction analysis
- Optional vote-weighted MAE

The detailed results are saved under `reports/`.

---

## Limitations

1. The 2022 election is used as a retrospective validation set and was also used during model comparison. It is therefore not a completely untouched final test set.
2. Annual demographic data cannot fully represent changes between the two elections held in 2019.
3. The model does not include campaign events, candidate changes, security events, turnout shocks, party mergers, or newly formed parties.
4. The relationship between demographic variables and voting patterns may change over time.
5. Some locality groups contain relatively few observations.
6. The model predicts locality-level bloc shares and not national seat allocation.
7. The current project is a proof of concept and is not yet a production election-forecasting system.

---

## Conclusions

The project demonstrates that locality-level demographic and socioeconomic information can explain a large part of the variation in historical voting patterns.

The strongest performance was achieved by combining:

- XGBoost
- CLR compositional modeling
- Segmentation by Arab/Non-Jewish locality type
- A Druze-majority indicator

The final validation result is an overall MAE of approximately **3.99 percentage points** and an R² of approximately **0.931**.

---

## Next Steps

1. Retrain the selected model structure using all available elections, including 2022.
2. Prepare a single 2024 demographic record for each locality.
3. Generate one future prediction per locality.
4. Evaluate the model on the next available election as a truly unseen test set.
5. Add temporal and political-event features.
6. Evaluate leave-one-election-out validation.
7. Move reusable functions from notebooks into `src/`.
8. Complete `data/data_dictionary.md`.
9. Add the final slide deck to `presentation/`.
10. Add a concise performance summary to `reports/model_results.md`.

---

---

