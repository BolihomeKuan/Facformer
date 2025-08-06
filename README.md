# Facformer: Deep Learning for Aquatic Products Shelf-Life Prediction

This repository contains the implementation of **Facformer**, a Transformer-based deep learning model specifically designed for predicting shelf-life of aquatic products under variable temperature conditions.

## Model Architecture

Facformer introduces two key innovations:

1. **Fac-Attention Mechanism**: Enables forward-looking perception of environmental conditions for enhanced temperature-aware predictions
2. **Mark Embedding Module**: Encodes dynamic temperature patterns to capture environmental context in cold-chain scenarios

## Key Features

- **Multi-stream Integration**: Four-stream fusion mechanism combining quality indicators, masked attention, convolutional features, and environmental streams
- **Temperature Robustness**: Specifically designed for variable temperature storage conditions
- **High Accuracy**: Achieves 99.2% accuracy under constant temperatures and 95.3% under variable conditions

## Installation

### Requirements
```
Python 3.8+
PyTorch 1.9+
NumPy
Pandas
Scikit-learn
```

### Setup
```bash
git clone [repository-url]
cd Facformer
pip install -r requirements.txt
```

## Usage

### Training
```bash
python run.py --model Facformer --data Food-SPV --data_path ./data/Food-SPV/
```

### Evaluation
```bash
python evaluate.py --results_path ./results/[experiment_name]/
```

### Demo
```bash
python demo.py --data_path ./data/Food-SPV/ --model_path ./checkpoints/[model_checkpoint]
```

## Model Components

### Core Files
- `models/Facformer.py`: Main model implementation
- `layers/Fac_Attention.py`: Fac-Attention mechanism
- `layers/Fac_Embed.py`: Mark Embedding module
- `layers/Facformer_EncDec.py`: Encoder-decoder architecture

### Data Processing
- `data_provider/data_loader.py`: Data loading and preprocessing utilities

## Data Format

The model expects time-series data with the following structure:
- Temperature readings (°C)
- Storage time (days)
- Quality indicators (species-specific metrics)
- Shelf-life labels

Data should be organized in the Food-SPV database format with separate directories for constant and variable temperature conditions.

**Data Availability**: The datasets used in this research are available through the Food-SPV database platform at http://42.193.99.162:3000. The platform provides comprehensive storage life data for various aquatic species including fish, shrimp, and shellfish under different temperature conditions.

## Contact

For questions about the implementation or access to the Food-SPV database, please contact the authors.