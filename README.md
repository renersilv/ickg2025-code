# Semantic and Structural Drift in Financial Knowledge Graphs: A Robustness Analysis of GNN-based Fraud Detectors

This repository contains the implementation of experiments analyzing the **robustness of Graph Neural Networks (GNNs) against semantic and structural drift** in financial Knowledge Graphs for fraud detection. The study provides a foundational analysis of how different GNN architectures degrade over time when exposed to natural data drift.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Framework Architecture](#framework-architecture)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Experiment Reproduction](#experiment-reproduction)
- [Results](#results)
- [Hardware Requirements](#hardware-requirements)
- [Citation](#citation)

## 🎯 Overview

### Research Problem

While Graph Neural Networks (GNNs) have achieved impressive performance in academic benchmarks for fraud detection, their real-world deployment is challenged by **data drift**, where the statistical properties of data change over time. This drift manifests as both **semantic drift** (changes in feature distributions) and **structural drift** (evolution of graph topology) in financial Knowledge Graphs.

### Our Research Focus: GNN Robustness Analysis

This study provides a **foundational empirical analysis** of GNN performance degradation under natural data drift in financial contexts. Unlike previous work that focuses on proposing new architectures, our goal is to **rigorously characterize and quantify** the drift problem across established GNN architectures.

**Research Hypotheses:**
- **H1**: GNN models trained for fraud detection suffer significant performance degradation when exposed to temporal data drift
- **H2**: Different GNN architectures (convolutional vs. attention-based) exhibit distinct patterns of robustness and degradation  
- **H3**: The nature of decay suggests that both semantic and structural changes in the graph contribute to model degradation

**Methodological Framework:**
- **Temporal Split Strategy**: 70% historical data for training, 30% for monitoring across 50 sequential windows
- **Multi-stage Feature Engineering**: 110 engineered features with robust normalization and anomaly detection
- **Heterogeneous Knowledge Graph**: 6 node types, 7 edge types, 292,616 nodes, 5.1M edges
- **GNN Architecture Comparison**: R-GCN, HGT, and HAN evaluation under identical conditions
- **Drift Quantification**: Performance monitoring across 50 temporal windows (1,704 transactions each)
- **MLOps Perspective**: Practical insights for production deployment and monitoring

**Supporting Framework: Multi-stage Feature Engineering**

The study employs a systematic **multi-stage feature engineering pipeline** to create robust features while preventing data leakage:

1. **Initial Selection and Pruning** - Iterative removal of 101 highly correlated features (>0.98), LightGBM-based ranking of remaining 332 features, selection of top 100
2. **Feature Creation and Enrichment** - Addition of temporal, monetary, interaction, and anomaly detection features (Isolation Forest)
3. **Robust Normalization** - Outlier clipping at 0.75%/99.25% percentiles followed by StandardScaler fitted only on training data

This results in **110 engineered features** per transaction, designed to capture complex fraud behaviors while maintaining temporal integrity.

### Key Contributions

- **Foundational Drift Analysis**: First rigorous comparative analysis quantifying performance degradation of foundational GNN architectures on financial KGs under natural drift
- **Robustness Methodology**: Strict "train once, monitor sequentially" protocol preventing data leakage across 50 temporal windows
- **Architecture Comparison**: Comprehensive evaluation of R-GCN (convolutional), HGT and HAN (attention-based) under identical conditions
- **Performance-Robustness Trade-off**: Demonstration that simpler architectures can be more robust than complex attention mechanisms
- **MLOps Framework**: Practical deployment strategy with dual-model systems and automated adaptation triggers
- **Drift Characterization**: Evidence for both semantic drift (threshold degradation) and structural drift (sudden performance drops)

## 📊 Dataset

**IEEE-CIS Fraud Detection Dataset**
- **Source**: IEEE Computational Intelligence Society and Vesta Corporation
- **Subset Used**: 284,000 continuous sequential transactions (computationally tractable)
- **Features**: 110 engineered features (from 433 original features)
- **Target**: Binary fraud classification (3.62% fraud rate)
- **Temporal Span**: Sequential transactions allowing granular drift analysis

### Data Characteristics
- **Class Distribution**: 96.38% legitimate, 3.62% fraudulent transactions
- **Feature Engineering**: Multi-stage pipeline with robust normalization
- **Temporal Integrity**: Strict chronological ordering maintained
- **Real-world Complexity**: Natural drift patterns, missing values, outliers

## 🏗️ Framework Architecture

### Robustness Analysis Pipeline

```
Sequential Transactions → Temporal Split → Feature Engineering → KG Construction → GNN Training → Drift Monitoring → Robustness Analysis
```

### Knowledge Graph Schema

**Node Types (6) - 292,616 total nodes:**
- `transaction` (284,000 nodes) - Individual financial transactions with 110 features
- `card` (8,169 nodes) - Credit card entities (fraud ring detection)
- `address` (161 nodes) - Geographical address entities
- `email` (108 nodes) - Email domain entities  
- `product` (5 nodes) - Product category entities
- `temporal` (173 nodes) - Discrete 6-hour time windows

**Edge Types (7) - 5,111,737 total edges:**
- `relates_to`: Transaction → Transaction (k-NN behavioral similarity)
- `uses_card`: Transaction → Card (card usage relationships)
- `at_address`: Transaction → Address (location relationships)
- `uses_email`: Transaction → Email (email domain relationships)
- `buys_product`: Transaction → Product (product category relationships)
- `in_timewindow`: Transaction → Temporal (temporal context)
- `sequential`: Temporal → Temporal (time sequence)

### GNN Architectures Tested

1. **R-GCN (Relational Graph Convolutional Network)**
   - **Architecture**: Convolutional approach with relation-specific transformations
   - **Hyperparameters**: 128 hidden channels, 3 layers, 0.4 dropout
   - **Key Strength**: Robust structural pattern learning

2. **HGT (Heterogeneous Graph Transformer)**
   - **Architecture**: Attention-based mechanism for heterogeneous graphs
   - **Hyperparameters**: 64 hidden channels, 3 layers, 8 heads, 0.6 dropout
   - **Key Strength**: Complex node and edge type interactions

3. **HAN (Heterogeneous Attention Network)**
   - **Architecture**: Hierarchical attention (node-level + semantic-level)
   - **Hyperparameters**: 64 hidden channels, 3 layers, 4 heads, 0.6 dropout
   - **Key Strength**: Multi-level attention aggregation

### Training Configuration

**Common Settings:**
- **Optimizer**: AdamW (better weight decay handling)
- **Learning Rate**: 0.002 with Cosine Annealing
- **Loss Function**: Focal Loss (α=0.95, γ=2.5) for class imbalance
- **Training**: 400 epochs with early stopping (patience=40)
- **Hardware**: NVIDIA Tesla T4 GPU

### Temporal Split Strategy

**Historical Set (70% - 198,800 transactions):**
- **Training Set (70%)**: Model training and optimization
- **Validation Set (15%)**: Hyperparameter tuning and early stopping  
- **Test Set (15%)**: Baseline performance establishment (T=0)

**Monitoring Set (30% - 85,200 transactions):**
- **50 Sequential Windows**: 1,704 transactions each (~12 hours of activity)
- **Strict Chronological Order**: No data leakage between sets
- **Real-world Simulation**: "Train once, monitor sequentially" protocol

## 🚀 Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 16+ GB RAM
- 50+ GB free disk space

### Environment Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd graphsentinel_2.0
```

2. **Create virtual environment:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### GPU Setup (Optional but Recommended)

For CUDA acceleration, ensure you have:
- NVIDIA GPU with CUDA 11.0+
- Compatible PyTorch installation
- torch-geometric with GPU support

## 📁 Project Structure

```
graphsentinel_2.0/
├── data/                          # Data storage
│   ├── raw/                      # Original IEEE-CIS dataset
│   ├── parquet/                  # Processed datasets
│   ├── graph/                    # Constructed graphs
│   └── statis/                   # Statistical analysis results
├── src/code/ieee-cis/            # Main experiment code
│   ├── 1_data_preparation/       # Data preprocessing
│   ├── 2_statistical_analysis/   # Multi-statistical analysis
│   ├── 3_feature_engineering/    # Feature creation and selection
│   └── 4_graph_knowledge_drift/  # Graph construction and GNN training
├── results/                      # Experiment results
├── logs/                         # Execution logs
└── images/                       # Generated visualizations
```

### Key Script Categories

| Directory | Scripts | Purpose |
|-----------|---------|---------|
| `1_data_preparation/` | `1_select_initial_features.py`<br>`2_preprocess_raw_data.py` | Initial feature selection<br>Data preprocessing and cleaning |
| `2_statistical_analysis/` | `1_run_all_analyses.py`<br>`statis/` modules | Orchestrates all statistical analyses<br>Individual analysis implementations |
| `3_feature_engineering/` | `1_create_feature_sets.py`<br>`2_feature_importance.py`<br>`3_normalization.py` | Feature set consolidation<br>Importance ranking<br>Data normalization |
| `4_graph_knowledge_drift/` | `1_build_graph.py`<br>`2_run_gnn_rgcn.py`<br>`3_run_gnn_hgt.py`<br>`4_run_gnn_han.py` | Graph construction<br>R-GCN training<br>HGT training<br>HAN training |

## 🔬 Experiment Reproduction

### Complete Pipeline Execution

Execute the experiments in sequence:

#### Step 1: Data Preparation
```bash
cd src/code/ieee-cis/1_data_preparation
python 1_select_initial_features.py
python 2_preprocess_raw_data.py
```

#### Step 2: Statistical Analysis
```bash
cd ../2_statistical_analysis
python 1_run_all_analyses.py
```

#### Step 3: Feature Engineering
```bash
cd ../3_feature_engineering
python 1_create_feature_sets.py
python 2_feature_importance.py
python 3_normalization.py
```

#### Step 4: Graph Construction and GNN Training
```bash
cd ../4_graph_knowledge_drift

# Build heterogeneous knowledge graph
python 1_build_graph.py

# Train GNN models
python 2_run_gnn_rgcn.py
python 3_run_gnn_hgt.py
python 4_run_gnn_han.py
```

### Configuration Parameters

#### Key Hyperparameters
- **Random Seed**: 42 (for reproducibility)
- **Train/Val/Test Split**: 60%/20%/20% (temporal split)
- **Graph Construction**:
  - k-NN neighbors: 10
  - Similarity threshold: 0.7
  - Temporal window: 24 hours
- **GNN Training**:
  - Epochs: 400
  - Learning rate: 0.002
  - Patience: 40
  - Batch processing: Full graph
  - Mixed precision: Enabled

#### Hardware Optimization
- **GPU Memory**: Automatic detection and optimization
- **CPU Fallback**: Automatic when GPU unavailable
- **Memory Management**: Garbage collection and optimization

### Expected Runtime

| Component | CPU | GPU (Tesla T4) |
|-----------|-----|---------------|
| Data Preparation | ~5 minutes | ~3 minutes |
| Statistical Analysis | ~15 minutes | ~8 minutes |
| Feature Engineering | ~10 minutes | ~5 minutes |
| Graph Construction | ~20 minutes | ~12 minutes |
| R-GCN Training | ~45 minutes | ~15 minutes |
| HGT Training | ~60 minutes | ~20 minutes |
| HAN Training | ~50 minutes | ~18 minutes |
| **Total Pipeline** | **~3.5 hours** | **~1.5 hours** |

## 📈 Results

### Baseline Performance (T=0)

Initial performance on the held-out test set before drift monitoring:

| Model | AUC-ROC | Precision | Recall | F1-Score |
|-------|---------|-----------|---------|----------|
| **R-GCN** | **0.990** | 0.822 | **0.913** | **0.865** |
| **HGT** | 0.984 | **0.807** | 0.696 | 0.747 |
| **HAN** | 0.982 | 0.773 | 0.771 | 0.772 |

### Performance Degradation Under Data Drift

**Key Findings from 50 Temporal Windows:**

#### R-GCN: Most Robust Architecture
- **Baseline F1-Score**: 0.865 → **Mean F1-Score**: 0.868 (stable)
- **Standard Deviation**: 0.046 (lowest volatility)
- **Performance Range**: 0.748 - 0.961
- **Robustness Profile**: Consistent high performance with minimal fluctuation

#### HGT: Highest Volatility  
- **Baseline F1-Score**: 0.747 → **Mean F1-Score**: 0.704 (degradation)
- **Standard Deviation**: 0.108 (highest volatility)
- **Performance Range**: 0.455 - 0.860
- **Degradation**: **40% relative drop** in worst case (0.747 → 0.455)

#### HAN: Moderate Stability
- **Baseline F1-Score**: 0.772 → **Mean F1-Score**: 0.724 (slight degradation)
- **Standard Deviation**: 0.091 (moderate volatility)
- **Performance Range**: 0.451 - 0.871
- **Pattern**: Balanced performance with moderate fluctuation

### Critical Drift Insights

1. **Performance-Robustness Trade-off**: R-GCN achieves both highest baseline performance AND highest stability
2. **Attention Mechanisms Vulnerable**: Complex attention-based models (HGT, HAN) show higher volatility than convolutional approaches
3. **Non-monotonic Decay**: Drift manifests as extreme, non-stationary volatility rather than smooth degradation
4. **AUC-ROC Stability**: Ranking metrics remain more stable than classification metrics, indicating semantic drift

### Evidence for Semantic and Structural Drift

**Semantic Drift Indicators:**
- **AUC-ROC Stability**: Ranking ability remains relatively stable across all models
- **Classification Threshold Degradation**: F1-Score, Precision, Recall show high volatility
- **Pattern**: Models can still rank fraudulent transactions but optimal decision thresholds shift

**Structural Drift Indicators:**  
- **Sharp Performance Drops**: Sudden, unpredictable performance degradation events
- **High Volatility**: Non-monotonic decay patterns characteristic of topology changes
- **Pattern**: Emergence of new fraud ring structures with novel topological patterns

### MLOps Framework for Production Deployment

Based on the drift analysis findings, the paper proposes a practical deployment strategy:

#### Phase 1: Dual-Model System
- **Primary Model**: High-performance architecture (HGT/HAN) for peak accuracy
- **Canary Model**: Stable architecture (R-GCN) as reliable fallback

#### Phase 2: Advanced Monitoring
- **Performance Monitoring**: Track F1-Score AND volatility over sliding windows
- **Divergence Monitoring**: Monitor prediction disagreement between primary and canary models

#### Phase 3: Automated Adaptation
- **Threshold Recalibration**: Lightweight adjustment when divergence increases
- **Model Fallback**: Switch to stable model when primary performance drops >15% for 3+ windows  
- **Shadow Retraining**: Heavy retraining triggered by significant drift signals

## 💻 Hardware Requirements

### Minimum Requirements
- **CPU**: 4 cores, 2.5+ GHz
- **RAM**: 16 GB
- **Storage**: 50 GB free space
- **Python**: 3.8+

### Recommended Requirements
- **CPU**: 8+ cores, 3.0+ GHz
- **RAM**: 32 GB
- **GPU**: NVIDIA with 8+ GB VRAM (Tesla T4, RTX 3080, or better)
- **Storage**: 100 GB SSD
- **CUDA**: 11.0+

### Tested Environments
- **Development**: Ubuntu 20.04, Python 3.12, Tesla T4
- **Production**: Linux servers with CUDA acceleration
- **Compatibility**: Windows 10/11, macOS (CPU-only)

## 🔧 Troubleshooting

### Common Issues

1. **GPU Memory Errors**
   - Reduce batch size or use gradient checkpointing
   - Enable mixed precision training

2. **Dependency Conflicts**
   - Use the exact versions in `requirements.txt`
   - Create fresh virtual environment

3. **Data Loading Issues**
   - Ensure sufficient RAM for dataset
   - Check file paths and permissions

4. **CUDA Compatibility**
   - Verify CUDA version compatibility
   - Install appropriate PyTorch version

### Performance Optimization

- **CPU**: Use `OMP_NUM_THREADS=4` for optimal threading
- **GPU**: Set `CUDA_LAUNCH_BLOCKING=1` for debugging
- **Memory**: Monitor with `nvidia-smi` and adjust accordingly

## 📚 Citation

If you use this work in your research, please cite:

```bibtex
@inproceedings{menezes2025semantic,
  title={Semantic and Structural Drift in Financial Knowledge Graphs: A Robustness Analysis of GNN-based Fraud Detectors},
  author={Menezes, Rener S. and Filho, Raimir H.},
  booktitle={Proceedings of the International Conference on Knowledge Graphs (ICKG)},
  year={2025},
  note={Submitted to ICKG 2025},
  organization={IEEE}
}
```

## 🏆 Key Research Contributions

1. **First Rigorous Drift Analysis**: Comprehensive evaluation of foundational GNN architectures under natural financial data drift
2. **Performance-Robustness Trade-off Discovery**: Demonstration that simpler convolutional architectures can outperform complex attention mechanisms in stability
3. **Practical MLOps Framework**: Actionable deployment strategy with dual-model systems and automated monitoring
4. **Methodological Rigor**: Strict temporal split preventing data leakage with 50-window monitoring protocol
5. **Foundational Benchmark**: Establishes baseline for future drift-aware GNN research in financial applications

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for guidelines.

## 📞 Contact

For questions and support:
- **Primary Author**: Rener S. Menezes (rener@edu.unifor.br)
- **Supervisor**: Raimir H. Filho (raimir@unifor.br)
- **Institution**: University of Fortaleza (UNIFOR), Brazil
- **Issues**: Use GitHub Issues
- **Documentation**: See `/docs` for detailed technical documentation

## 🙏 Acknowledgments

This work was supported by **FITBANK PAGAMENTOS ELETRONICOS SA**, funding the Master's studies of Rener S. Menezes at the University of Fortaleza.

---

**Semantic and Structural Drift Analysis** - Advancing robust GNN deployment through foundational drift characterization in financial Knowledge Graphs.
