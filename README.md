# CEO Recommender

A Two Tower Recommender System for CEO-firm matching.

## Overview

This project implements a Two Tower neural network architecture for building a recommender system. The Two Tower model is a deep learning approach that separately encodes query (user) and candidate (item) features into embedding spaces, then computes similarity scores between them.

## Two Tower Architecture

The Two Tower model consists of:

1. **Query Tower**: Encodes firm/context features into dense embeddings
2. **Candidate Tower**: Encodes CEO features into dense embeddings
3. **Similarity Layer**: Computes dot product or cosine similarity between the two towers

This architecture enables efficient retrieval by pre-computing candidate embeddings and performing approximate nearest neighbor search at inference time.

## Features

- Separate neural networks for query and candidate encoding
- Efficient candidate retrieval using embedding similarity
- Scalable architecture for large-scale recommendation tasks

## Getting Started

(Add installation and usage instructions here)

## References

- Based on the Two Tower architecture commonly used in industrial recommendation systems
- Implements principles from modern deep learning-based information retrieval

## License

(Add license information)
