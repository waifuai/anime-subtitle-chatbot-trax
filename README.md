# Anime Subtitle Chatbot with Trax

A Trax implementation of a Transformer-based chatbot trained on the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles). Generates contextual responses to anime dialogue inputs.

## Installation

```bash
pip install -e .
```

## Dataset Preparation

1. Download dataset from Kaggle and place raw text files in `local_data/data/`
2. Format data into parallel input/target pairs:
   - `input.txt`: One input phrase per line
   - `output.txt`: Corresponding response phrases
3. Preprocess data:
```bash
chmod +x scripts/prep.sh
./scripts/prep.sh
```

## Training

Start training with Transformer model:
```bash
./scripts/train.sh
```
- Model checkpoints saved to `local_data/model/`
- Training logs: `train.log`

## Prediction

Generate responses interactively:
```bash
./scripts/predict.sh --decode_interactive
```

Batch decode from file:
```bash
./scripts/predict.sh --decode_from_file=phrases_input.txt
```

## Project Structure

```
.
├── anime_chatbot/              # Core module
│   ├── trainer/                # Training logic
│   │   ├── problem.py          # Dataset/model configuration
│   ├── __init__.py
├── scripts/
│   ├── train.sh                # Launch training
│   ├── predict.sh              # Run inference
│   ├── prep.sh                 # Data preprocessing
└── local_data/                 # Data/model storage
    ├── data/                   # Processed datasets
    └── model/                  # Trained models
```

## Model Architecture

**Transformer Configuration** (`problem.py`):
- 2 hidden layers
- 128-dimensional embeddings
- 4 attention heads
- 512 feed-forward dimension
- 0.6 dropout rate
- Learning rate: 0.05

Adapted from Google's Tensor2Tensor tutorial with Trax optimizations.

## Acknowledgements

Based on Tensor2Tensor's [text generation tutorial](https://cloud.google.com/blog/products/gcp/cloud-poetry-training-and-hyperparameter-tuning-custom-text-models-on-cloud-ml-engine), converted to Trax for improved performance and simplicity.