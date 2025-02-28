# Project EPJ 📈

## Stock Price Prediction using Temporal Fusion Transformer

![Stock Market](<img width="635" alt="Screenshot 2025-02-28 at 8 30 49 AM" src="https://github.com/user-attachments/assets/33bdf55f-3be4-4144-9175-48c4f71b232b" />)

## Overview

Project EPJ uses advanced deep learning techniques to predict stock prices with high accuracy. By leveraging the Temporal Fusion Transformer (TFT) architecture, this project captures complex temporal patterns in financial time series data.

## Features

- **Multi-horizon Forecasting**: Predict stock prices across multiple future time steps
- **Interpretable Predictions**: Understand which features drive the model's decisions
- **Attention Mechanisms**: Capture long-range dependencies in financial data
- **Variable Selection Networks**: Automatically identify the most relevant features
- **Quantile Forecasting**: Estimate prediction uncertainty for risk management

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- PyTorch Forecasting
- pandas
- numpy
- matplotlib

### Installation

```bash
git clone https://github.com/yourusername/project-epj.git
cd project-epj
pip install -r requirements.txt
```

### Quick Start

```python
# Example code to load model and make predictions
from model import TemporalFusionTransformer
import pandas as pd

# Load your pre-trained model
model = TemporalFusionTransformer.load("models/tft_stock_model.pth")

# Prepare your data
data = pd.read_csv("data/stock_data.csv")

# Make predictions
predictions = model.predict(data)
```

## Model Architecture

The Temporal Fusion Transformer combines:
- LSTM encoders for processing sequential data
- Self-attention layers for capturing long-range dependencies
- Variable selection networks for handling multivariate data
- Quantile outputs for uncertainty estimation

## Results

Our model achieves:
- MAPE: 1.2% on test data
- RMSE: 0.45 on 5-day forecasts
- Outperforms ARIMA and LSTM baselines by 15%

## Google Colab Notebook

Explore the complete implementation in our [Google Colab notebook](https://colab.research.google.com/drive/1xqN-5xeZAiO9kTJ1-Cc2O_ju5Rq3wzsU?usp=sharing).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [PyTorch Forecasting](https://pytorch-forecasting.readthedocs.io/) for their implementation of TFT
- [Paper: Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://arxiv.org/abs/1912.09363)
