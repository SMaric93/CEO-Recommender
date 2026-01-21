"""
Two-Towers Neural Network Analysis Module.

Provides analysis for the Two-Tower recommender model:
- Train/evaluate pipeline with robustness
- Embedding visualization (t-SNE/UMAP)
- Interaction heatmaps
- Hyperparameter sensitivity
- Counterfactual analysis
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class TwoTowersResult:
    """Container for Two-Towers analysis results."""
    config_name: str
    n_train: int
    n_val: int
    train_loss: float
    val_loss: float
    train_corr: float
    val_corr: float
    embedding_dim: int
    ceo_embedding: Optional[np.ndarray] = None
    firm_embedding: Optional[np.ndarray] = None
    model: Any = None


class TwoTowersAnalyzer:
    """
    Two-Towers model analyzer with robustness and visualization.
    
    Example usage:
        from ceo_firm_matching import Config, DataProcessor
        analyzer = TwoTowersAnalyzer(df, config)
        results = analyzer.run_training()
        heatmap = analyzer.compute_interaction_heatmap('logatw', 'Age')
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        config: Optional[Any] = None,
        device: Optional[str] = None
    ):
        """
        Initialize Two-Towers analyzer.
        
        Args:
            df: DataFrame with CEO-Firm features
            config: Config object (uses default if None)
            device: PyTorch device ('cpu', 'mps', 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required. Install: pip install torch")
        
        self.df = df.copy()
        self.results: Dict[str, TwoTowersResult] = {}
        
        # Import package components
        from ceo_firm_matching import Config, DataProcessor, CEOFirmDataset, CEOFirmMatcher, train_model
        
        self.Config = Config
        self.DataProcessor = DataProcessor
        self.CEOFirmDataset = CEOFirmDataset
        self.CEOFirmMatcher = CEOFirmMatcher
        self.train_model = train_model
        
        # Setup config
        if config is None:
            self.config = Config()
        else:
            self.config = config
        
        # Device
        if device:
            self.config.DEVICE = device
        
        print(f"📊 Two-Towers Analyzer: {len(df)} samples, device={self.config.DEVICE}")
    
    def _prepare_data(self, val_split: float = 0.2) -> Tuple[DataLoader, DataLoader, Any]:
        """Prepare data loaders."""
        from sklearn.model_selection import train_test_split
        
        processor = self.DataProcessor(self.config)
        df_clean = processor.prepare_features(self.df)
        
        train_df, val_df = train_test_split(df_clean, test_size=val_split, random_state=42)
        
        processor.fit(train_df)
        train_data = processor.transform(train_df)
        val_data = processor.transform(val_df)
        
        train_dataset = self.CEOFirmDataset(train_data)
        val_dataset = self.CEOFirmDataset(val_data)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)
        
        return train_loader, val_loader, processor, train_data
    
    def run_training(self, config_name: str = "default") -> TwoTowersResult:
        """Run training with current config."""
        print(f"🏗️ Training Two-Towers ({config_name})...")
        
        train_loader, val_loader, processor, train_data = self._prepare_data()
        
        # Train model
        model = self.train_model(train_loader, val_loader, train_data, self.config)
        
        if model is None:
            print("   ⚠️ Training failed")
            return None
        
        # Evaluate
        model.eval()
        train_preds, train_targets = [], []
        val_preds, val_targets = [], []
        
        with torch.no_grad():
            for batch in train_loader:
                pred = model(batch).cpu().numpy()
                train_preds.extend(pred)
                train_targets.extend(batch['target'].cpu().numpy())
            
            for batch in val_loader:
                pred = model(batch).cpu().numpy()
                val_preds.extend(pred)
                val_targets.extend(batch['target'].cpu().numpy())
        
        train_corr = np.corrcoef(train_preds, train_targets)[0, 1]
        val_corr = np.corrcoef(val_preds, val_targets)[0, 1]
        
        result = TwoTowersResult(
            config_name=config_name,
            n_train=len(train_preds),
            n_val=len(val_preds),
            train_loss=0.0,  # TODO: extract from training
            val_loss=0.0,
            train_corr=train_corr,
            val_corr=val_corr,
            embedding_dim=self.config.EMBEDDING_DIM,
            model=model
        )
        
        self.results[config_name] = result
        print(f"   ✓ Train corr={train_corr:.4f}, Val corr={val_corr:.4f}")
        
        return result
    
    def run_hyperparameter_robustness(self) -> Dict[str, TwoTowersResult]:
        """Run hyperparameter sensitivity analysis."""
        print("🔧 Two-Towers Hyperparameter Robustness...")
        
        results = {}
        
        # Learning rate sensitivity
        for lr in [1e-4, 1e-3, 1e-2]:
            self.config.LEARNING_RATE = lr
            name = f"lr_{lr}"
            result = self.run_training(name)
            if result:
                results[name] = result
        
        # Reset to default
        self.config.LEARNING_RATE = 1e-3
        
        # Embedding dimension sensitivity
        for dim in [32, 64, 128]:
            self.config.EMBEDDING_DIM = dim
            name = f"emb_{dim}"
            result = self.run_training(name)
            if result:
                results[name] = result
        
        # Reset
        self.config.EMBEDDING_DIM = 64
        
        return results
    
    def compute_interaction_heatmap(
        self,
        model: Any,
        processor: Any,
        firm_feature: str,
        ceo_feature: str,
        n_grid: int = 20
    ) -> pd.DataFrame:
        """
        Compute interaction heatmap between firm and CEO features.
        
        Args:
            model: Trained Two-Towers model
            processor: Fitted DataProcessor
            firm_feature: Firm feature for x-axis
            ceo_feature: CEO feature for y-axis
            n_grid: Grid resolution
            
        Returns:
            DataFrame with heatmap values
        """
        if processor.processed_df is None:
            print("   ⚠️ Processor not fitted with data")
            return pd.DataFrame()
        
        df = processor.processed_df
        
        # Create grid
        firm_range = np.linspace(df[firm_feature].min(), df[firm_feature].max(), n_grid)
        ceo_range = np.linspace(df[ceo_feature].min(), df[ceo_feature].max(), n_grid)
        
        # Template row (median values)
        template = df.median()
        
        # Compute predictions
        heatmap = np.zeros((n_grid, n_grid))
        model.eval()
        
        for i, ceo_val in enumerate(ceo_range):
            for j, firm_val in enumerate(firm_range):
                row = template.copy()
                row[firm_feature] = firm_val
                row[ceo_feature] = ceo_val
                
                # Create batch
                batch = processor._create_single_batch(row)
                
                with torch.no_grad():
                    pred = model(batch).item()
                
                heatmap[i, j] = pred
        
        return pd.DataFrame(
            heatmap,
            index=[f"{ceo_feature}={v:.2f}" for v in ceo_range],
            columns=[f"{firm_feature}={v:.2f}" for v in firm_range]
        )
    
    def extract_embeddings(self, model: Any, processor: Any) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract CEO and Firm embeddings from trained model.
        
        Returns:
            Tuple of (CEO embeddings, Firm embeddings)
        """
        model.eval()
        
        train_loader, _, _, _ = self._prepare_data()
        
        ceo_embs = []
        firm_embs = []
        
        with torch.no_grad():
            for batch in train_loader:
                ceo_emb = model.get_ceo_embedding(batch)
                firm_emb = model.get_firm_embedding(batch)
                ceo_embs.append(ceo_emb.cpu().numpy())
                firm_embs.append(firm_emb.cpu().numpy())
        
        return np.vstack(ceo_embs), np.vstack(firm_embs)
    
    def reduce_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = 'tsne',
        n_components: int = 2
    ) -> np.ndarray:
        """
        Reduce embedding dimensionality for visualization.
        
        Args:
            embeddings: High-dimensional embeddings
            method: 'tsne' or 'pca'
            n_components: Output dimensions
            
        Returns:
            Reduced embeddings
        """
        if not SKLEARN_AVAILABLE:
            print("   ⚠️ scikit-learn required for embedding reduction")
            return embeddings[:, :n_components]
        
        if method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=n_components)
        
        return reducer.fit_transform(embeddings)
    
    def get_comparison_table(self) -> pd.DataFrame:
        """Generate model comparison table."""
        rows = []
        for name, r in self.results.items():
            rows.append({
                'Config': name,
                'N_train': r.n_train,
                'N_val': r.n_val,
                'Emb_dim': r.embedding_dim,
                'Train_corr': f"{r.train_corr:.4f}",
                'Val_corr': f"{r.val_corr:.4f}",
            })
        return pd.DataFrame(rows)
    
    def to_latex(self, title: str = "Two-Towers Model Comparison", label: str = "tab:tt") -> str:
        """Generate LaTeX comparison table."""
        df = self.get_comparison_table()
        
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            f"\\caption{{{title}}}",
            f"\\label{{{label}}}",
            r"\begin{tabular}{lccccc}",
            r"\hline\hline",
            r"Config & N (train) & N (val) & Emb. Dim & Train Corr & Val Corr \\",
            r"\hline",
        ]
        
        for _, row in df.iterrows():
            lines.append(
                f"{row['Config']} & {row['N_train']} & {row['N_val']} & "
                f"{row['Emb_dim']} & {row['Train_corr']} & {row['Val_corr']} \\\\"
            )
        
        lines.extend([
            r"\hline\hline",
            r"\end{tabular}",
            r"\end{table}",
        ])
        
        return "\n".join(lines)
