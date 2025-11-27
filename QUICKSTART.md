# Quickstart Guide

This guide will help you set up the environment and run the Two Tower CEO-Firm Matching model.

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

## 3. Run the Model
To train the model and generate visualizations (Interaction Heatmap, PDP Plots, SHAP Summary):

```bash
python two_towers.py
```

### What to Expect
1.  **Data Loading**: The script will load `Data/ceo_types_v0.2.csv`.
2.  **Training**: It will train the Two Tower neural network for 20 epochs, printing the Weighted MSE Loss.
3.  **Visualizations**:
    *   **Interaction Heatmap**: A window showing the interaction between Firm Size (Log Assets) and CEO Age.
    *   **PDP Plots**: Partial Dependence Plots for key features like Tenure, R&D, etc.
    *   **SHAP Summary**: A summary plot showing feature importance and impact.

*(Note: You may need to close each plot window to proceed to the next one or to finish the script.)*
