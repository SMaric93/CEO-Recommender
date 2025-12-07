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
