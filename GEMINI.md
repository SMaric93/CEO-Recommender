# Gemini Context: CEO-Firm Matching Two Tower Recommender

## Project Overview
This project implements a **Two Tower neural network architecture** for matching CEOs (candidates) with Firms (queries). It is designed to learn dense embeddings for both entities and compute similarity scores to predict match quality. The system uses PyTorch and includes specific optimizations for Apple Silicon (MPS).

## Architecture
The model follows the standard Two Tower paradigm:
1.  **Query Tower (Firm):** Encodes firm characteristics (financials, governance, investment, context) into a dense embedding.
2.  **Candidate Tower (CEO):** Encodes CEO attributes (demographics, education, skills, context) into a dense embedding.
3.  **Similarity Layer:** Computes the dot product (or cosine similarity) between the two embeddings to predict the `match_means` score.

## Key Files & Directories

*   **`two_towers.py`**: The core application. Contains:
    *   `Config`: Centralized configuration (paths, hyperparameters, feature definitions).
    *   `DataProcessor`: Handles loading, cleaning, feature engineering, and scaling.
    *   Model definitions (Towers) and training loop (inferred).
*   **`synthetic_data.py`**: Utility to generate synthetic datasets for testing the pipeline without real data.
*   **`setup_env.sh`**: Bash script for automated environment setup (venv creation, dependency installation).
*   **`requirements.txt`**: Python dependencies (`torch`, `pandas`, `scikit-learn`, `shap`, etc.).
*   **`Data/`**: Expected location for the input dataset (`ceo_types_v0.2.csv`).
*   **`Output/`**: Destination for generated visualizations (SVG format).

## Setup & Installation

The project uses a standard Python virtual environment.

1.  **Automated Setup:**
    ```bash
    ./setup_env.sh
    ```
2.  **Manual Activation:**
    ```bash
    source venv/bin/activate
    ```

## Usage

### Running the Model
*   **With Real Data:**
    ```bash
    python two_towers.py
    ```
    *Requires `Data/ceo_types_v0.2.csv` to exist and match the schema.*

*   **With Synthetic Data:**
    ```bash
    python two_towers.py --synthetic
    ```
    *Generates random data on the fly. Useful for debugging pipeline issues.*

## Data Configuration
The `Config` class in `two_towers.py` defines the expected schema.

*   **Firm Features:** `logatw` (Size), `exp_roa` (Performance), `boardindpw` (Governance), `rdintw` (R&D), etc.
*   **CEO Features:** `Age`, `Gender`, `maxedu` (Education), `Output`/`Throghput`/`Peripheral` (Skills).
*   **Target:** `match_means` (Match Quality).
*   **Weights:** `sd_match_means`.

## Outputs
The script is non-interactive and saves results to `Output/`:
*   **Heatmaps:** `heatmap_size_age.svg`, `heatmap_size_skill.svg`, etc. (Interaction effects).
*   **Partial Dependence Plots:** `pdp_plots.svg` (Marginal effects of specific features).
*   **SHAP Summary:** `shap_summary.svg` (Feature importance).

## Development Conventions
*   **Configuration:** All constants (hyperparameters, column names) are grouped in the `Config` class. **Modify this class to change model behavior.**
*   **State Management:** `DataProcessor` encapsulates encoders and scalers to ensure consistency between training and inference (or analysis).
*   **Hardware Acceleration:** The code checks for `mps` (Apple Silicon) and `cuda` (NVIDIA) availability automatically.

---

# Structural Distillation Network

## Overview
A second model architecture that distills BLM (Bonhomme-Lamadon-Manresa) econometric estimates into a neural network. Unlike the Two Tower model which predicts `match_means` directly, this model:

1. **Predicts Type Probabilities:** Learns to classify CEOs into 5 types and Firms into 5 classes
2. **Uses Frozen Interaction Matrix:** The BLM-estimated 5×5 interaction matrix is frozen (not learned)
3. **Computes Expected Match:** Match value = P(CEO type) × A × P(Firm class)

## Architecture
```
Observables → [CEO Tower] → P(CEO Type | X)
                                    ↓
                              A (Frozen BLM Matrix)
                                    ↓
Observables → [Firm Tower] → P(Firm Class | X)
                                    ↓
                            Expected Match Value
```

## Key Modules

| Module | Description |
|--------|-------------|
| `structural_config.py` | `StructuralConfig` dataclass with BLM priors |
| `structural_data.py` | `StructuralDataProcessor` for probability targets |
| `structural_model.py` | `StructuralDistillationNet` with frozen interaction matrix |
| `structural_training.py` | KL divergence training loop |
| `structural_explain.py` | `IlluminationEngine` for gradient sensitivity analysis |
| `structural_cli.py` | Command-line interface |

## Usage

### Running the Model
*   **With Real Data:**
    ```bash
    python structural_distillation_network.py
    ```
    *Requires `Data/blm_posteriors.csv` with probability columns.*

*   **With Synthetic Data:**
    ```bash
    python structural_distillation_network.py --synthetic
    ```

*   **Using Package CLI:**
    ```bash
    python -m ceo_firm_matching.structural_cli --synthetic --epochs 100
    ```

## Data Requirements
The Structural Distillation Network requires BLM posterior probabilities:

*   **CEO Probability Columns:** `prob_ceo_1`, `prob_ceo_2`, ..., `prob_ceo_5`
*   **Firm Probability Columns:** `prob_firm_1`, `prob_firm_2`, ..., `prob_firm_5`
*   **Observable Features:** Same as Two Tower model (Age, tenure, firm financials, etc.)

## Outputs
Results are saved to `Output/Structural_Distillation/`:
*   `interaction_matrix.png` - Visualization of frozen BLM A matrix
*   `match_drivers.png` - Gradient sensitivity bar chart
*   `sensitivity_analysis.csv` - Full sensitivity results
*   `type_distributions.csv` - CEO/Firm type probability statistics

## BLM Interaction Matrix
The default 5×5 matrix can be updated in `StructuralConfig.BLM_INTERACTION_MATRIX`:
```python
# Example: Replace with Table 3 estimates from BLM paper
BLM_INTERACTION_MATRIX = [
    [-0.5, -0.3,  0.0,  0.1,  0.2],  # CEO Type 1
    [-0.2, -0.1,  0.1,  0.3,  0.4],  # CEO Type 2
    [ 0.0,  0.2,  0.4,  0.6,  0.7],  # CEO Type 3
    [ 0.1,  0.4,  0.7,  0.9,  1.1],  # CEO Type 4
    [ 0.3,  0.6,  0.9,  1.2,  1.5],  # CEO Type 5 (Star)
]
```

