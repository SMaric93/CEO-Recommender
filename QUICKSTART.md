# Quickstart Guide

This guide will help you set up the environment, configure your data, and run the Two Tower CEO-Firm Matching model.

## Prerequisites
- Python 3.8 or higher
- A terminal (bash/zsh)

## 1. Automated Setup
We have provided a script to automatically create a virtual environment, install dependencies, and set up the local package.

Run the following command in your terminal:

```bash
./setup_env.sh
```

## 2. Activate Environment
Once the setup is complete, activate the virtual environment:

```bash
source venv/bin/activate
```

## 3. Command Line Usage

### Option A: Run with Real Data (Default)
To train the model using your dataset (see "Data Requirements" below):

```bash
python two_towers.py
```

### Option B: Run with Synthetic Data
To verify the pipeline using generated synthetic data (useful for testing without the dataset):

```bash
python two_towers.py --synthetic
```

## 4. Data Requirements
If running with real data (Option A), the script expects a CSV file at `Data/ceo_types_v0.2.csv`.

The CSV **must** contain the following columns:

### Identifiers
- `gvkey`: Firm identifier
- `match_exec_id`: Executive identifier

### Firm Features
- **Financials**: `logat` (Log Assets), `exp_roa` (Expected ROA), `ind_firms_60` (Industry size), `non_competition_score`
- **Governance (Board)**: `boardindpw` (Independence), `boardsizew` (Size), `busyw` (Busy Directors), `pct_blockw` (Blockholder %)
- **Investment**: `xrd_int` (R&D Intensity), `capx_int` (Capital Expenditure Intensity)
- **Context**: `compindustry` (Industry category), `fiscalyear`

### CEO Features
- **Demographics**: `Age`, `Gender`, `DOB` (Date of Birth), `year_born`
- **Education**: `maxedu` (Max Education Level), `ivy` (Ivy League Dummy)
- **Skills/Types**: `Output`, `Throghput`, `Peripheral`
- **Context**: `ceo_year`, `dep_baby_ceo`

### Targets (for Training)
- `match_means`: The match quality score (target variable)
- `sd_match_means`: Standard deviation of the match mean (used for weighting)

## 5. Outputs & Visualizations
The script does **not** open interactive windows. Instead, all visualizations are automatically saved to the `Output/` directory as SVG files.

### Generated Files (`Output/` folder):

#### 1. Interaction Heatmaps
Visualizes how match quality changes when varying two features simultaneously.
-   **`heatmap_size_age.svg`**: Firm Size (`logat`) vs CEO Age.
-   **`heatmap_size_skill.svg`**: Firm Size (`logat`) vs CEO Skill (`Output`).
-   **`heatmap_perf_exp.svg`**: Firm Performance (`exp_roa`) vs CEO Experience (`tenure`).

#### 2. Partial Dependence Plots (PDP)
-   **`pdp_plots.svg`**: Shows the marginal effect of specific features on the predicted match score.
-   **Included Features**:
    -   **CEO**: Age, Output, Throughput, Tenure.
    -   **Firm**: Log Assets, Exp ROA, Non-competition Score.
    -   **Board**: Board Independence, Board Size, Busy Directors, Blockholder %.
    -   **Investment**: R&D Intensity, Capital Expenditure Intensity.

#### 3. SHAP Summary (Optional)
-   **`shap_summary.svg`**: (If enabled in code) A global summary of feature importance using SHAP values.

---
*Note: If running on macOS with Apple Silicon (M1/M2/M3), the script is optimized to handle MPS (Metal Performance Shaders) limitations automatically.*
