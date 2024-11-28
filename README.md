# Dependency Chain Optimization Using Machine Learning 🔗

This project implements an advanced system for optimizing dependency chains in software projects using a combination of graph-based analysis, machine learning, and reinforcement learning. The system uses a critical node classifier and a reinforcement learning agent to generate optimized dependency chains.

## 🌟 Features

- Graph-based representation of dependencies using networkx
- Critical node classification using random forest
- Reinforcement learning for chain optimization
- Multi-objective reward function incorporating security, performance, and freshness
- Curriculum learning approach for training

## 📊 System Architecture

The system consists of two main components:
1. Critical Node Classifier
2. Reinforcement Learning Agent

### Key Statistics
- Total nodes: 442,275
- Critical nodes: 25,072
- Non-critical nodes: 417,203

## 🚀 Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

Required packages:
- networkx
- numpy
- pandas
- scikit-learn
- torch
- gym

### ⚠️ Important: Execution Order

The project must be executed in the following order:

1. First, run the critical node classifier:
```bash
python critical.py
```
This will:
- Process the dependency graph
- Extract features
- Train the random forest classifier
- Save the model as a pickle file

2. Then, run the reinforcement learning component:
```bash
python reinforcement.py
```
This will:
- Load the saved classifier model
- Initialize the RL environment
- Train the agent using curriculum learning
- Generate optimized dependency chains

## 🏗️ System Components

### Critical Node Classification
- Uses both topological and semantic features
- Features include:
  - Degree centrality
  - Betweenness centrality
  - PageRank
  - Clustering coefficient
  - Local risk ratio
  - Node types
  - Dependency scopes
  - Quality metrics

### Reinforcement Learning Environment
- Custom OpenAI Gym environment
- State space includes current node features, chain metrics, and quality indicators
- Action space for node selection with constraints
- Multi-objective reward function

## 📈 Performance

Classification Results:
- Non-Critical Class: 1.00 (Precision, Recall, F1-Score)
- Critical Class: 1.00 (Precision, Recall, F1-Score)
- Overall Accuracy: 1.00

Chain Quality Metrics:
- Average security score: 1.0
- Average performance score: 0.5
- Average freshness score: 0.5
- Valid chain ratio: 100%

## 🛠️ Future Improvements

Planned enhancements:
- Transformer-based feature processing
- Multi-agent approaches
- Temporal dependency patterns
- Compatibility scores
- Usage statistics integration
- Prioritized sampling
- Meta-learning approaches

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For any questions or feedback, please open an issue in the GitHub repository.
