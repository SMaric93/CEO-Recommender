# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a Two Tower Recommender System for CEO-firm matching. The architecture uses separate neural networks to encode firm features (Query Tower) and CEO features (Candidate Tower), then computes similarity scores between embeddings for matching recommendations.

### Two Tower Architecture Components

1. **Query Tower**: Encodes firm/context features into dense embeddings
2. **Candidate Tower**: Encodes CEO features into dense embeddings  
3. **Similarity Layer**: Computes dot product or cosine similarity between towers

The key advantage is efficient retrieval through pre-computed candidate embeddings and approximate nearest neighbor search at inference time.

## Development Commands

This is a new repository - implementation is pending. Once code is added, this section should include:

- How to set up the development environment (e.g., `pip install -r requirements.txt` or `poetry install`)
- How to run training scripts
- How to evaluate models
- How to run inference/predictions
- How to run tests (e.g., `pytest` or `python -m unittest`)
- How to format/lint code (e.g., `black .`, `ruff check`, `mypy`)

## Project Context

This project is part of BLM (Bonhomme-Lamadon-Manresa) replication research on complementarities, focusing on CEO-firm matching using modern deep learning techniques.

### Key Research Concepts

- **CEO-Firm Matching**: Pairing executives with companies based on complementary features
- **Two Tower Model**: Industry-standard architecture for large-scale recommendation systems
- **Embedding-based Retrieval**: Efficient similarity search in high-dimensional spaces

## Architecture Guidelines

When implementing the Two Tower model:

- Keep Query and Candidate towers independent during training to enable separate embedding computation
- Design towers to output same-dimensional embeddings for compatibility
- Consider using batch normalization or layer normalization for stable training
- Implement negative sampling strategies for training (e.g., in-batch negatives, hard negative mining)
- Use temperature-scaled softmax for training loss
- Enable efficient serving by pre-computing and indexing candidate embeddings

### Data Considerations

- Firm features may include: size, industry, performance metrics, location, historical patterns
- CEO features may include: experience, education, track record, management style indicators
- Consider feature engineering for both categorical and numerical inputs
- Handle missing data appropriately given the economic research context

## Related Work

This implementation relates to broader BLM replication work. Cross-reference with:
- Main BLM replication codebase
- Complementarities research paper implementations
- Related economic analysis code in parent directories
