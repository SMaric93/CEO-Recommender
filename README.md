# CEO-Firm Matching

A neural network recommendation system for CEO-Firm matching, featuring two model architectures:

1. **Two Tower Model** - Embedding-based similarity matching
2. **Structural Distillation Network** - BLM econometric prior integration

## Installation

```bash
# Clone and setup
cd "Two Towers Implementation"
./setup_env.sh

# Or manual setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### Two Tower Model

```bash
# With real data
python -m ceo_firm_matching.cli

# With synthetic data
python -m ceo_firm_matching.cli --synthetic
```

### Structural Distillation Network

```bash
# With synthetic data
python structural_distillation_network.py --synthetic

# With options
python -m ceo_firm_matching.structural_cli --synthetic --epochs 100 --batch-size 128
```

## Architecture

### Two Tower Model

```
Firm Features → [Firm Tower] → Firm Embedding
                                     ↓
                              Cosine Similarity → Match Score
                                     ↑
CEO Features  → [CEO Tower]  → CEO Embedding
```

### Structural Distillation Network

```
Firm Features → [Firm Tower] → P(Firm Class)
                                     ↓
                              A (Frozen BLM Matrix)
                                     ↓
CEO Features  → [CEO Tower]  → P(CEO Type)
                                     ↓
                            Expected Match Value
```

## Package Structure

```
ceo_firm_matching/
├── config.py              # Two Tower configuration
├── data.py                # Data processing
├── model.py               # CEOFirmMatcher
├── training.py            # Training loop
├── explain.py             # SHAP/PDP explainability
├── visualization.py       # Interaction heatmaps
├── cli.py                 # Two Tower CLI
├── structural_config.py   # BLM priors configuration
├── structural_data.py     # Probability target processing
├── structural_model.py    # StructuralDistillationNet
├── structural_training.py # KL divergence training
├── structural_explain.py  # Gradient sensitivity
├── structural_visualization.py # Type distribution plots
└── structural_cli.py      # Structural CLI
```

## Data Requirements

### Two Tower Model
- `Data/ceo_types_v0.2.csv` with `match_means` target

### Structural Distillation
- `Data/blm_posteriors.csv` with `prob_ceo_1..5` and `prob_firm_1..5`

## Outputs

Results saved to `Output/`:
- **Two Tower**: `heatmap_*.svg`, `pdp_plots.svg`, `shap_summary.svg`
- **Structural**: `interaction_matrix.png`, `match_drivers.png`, `type_distributions.csv`

## Testing

```bash
pytest tests/ -v
```

## License

MIT

