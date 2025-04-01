# Anime Subtitle Chatbot with Hugging Face Transformers

A chatbot fine-tuned on the [Anime Subtitles Dataset](https://www.kaggle.com/jef1056/anime-subtitles) using Hugging Face `transformers` and the `distilgpt2` model. Generates contextual responses to anime dialogue inputs.

This project was migrated from an older `trax` implementation due to dependency issues.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/waifuai/anime-subtitle-chatbot-trax # Or your repo URL
    cd anime-subtitle-chatbot-trax
    ```
2.  **Create and activate a virtual environment:** (Recommended)
    ```bash
    # Using venv
    python -m venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows

    # Or using uv (if installed)
    python -m uv venv .venv
    source .venv/bin/activate # Linux/macOS
    # .venv\Scripts\activate # Windows
    ```
3.  **Install dependencies:**
    ```bash
    # Using uv (recommended if using uv venv)
    python -m uv pip install -e ./src/anime_chatbot[test]

    # Or using pip
    pip install -e ./src/anime_chatbot[test]
    ```
    *Note: This will install PyTorch. Ensure you have the correct version (CPU or CUDA) for your system. Refer to the [PyTorch installation guide](https://pytorch.org/get-started/locally/) if needed.*

## Dataset Preparation

1.  Download the dataset from Kaggle (or use your own paired input/response data). Place the raw text files named `input.txt` (one input phrase per line) and `output.txt` (corresponding response phrases) in the `src/local_data/data/` directory.
2.  Run the data preparation script. This combines the input and output files into a single `train.txt` file formatted for fine-tuning `distilgpt2`.
    ```bash
    python src/scripts/prep_data.py
    ```
    This script will:
    *   Load the `distilgpt2` tokenizer.
    *   Add a special separator token (`<|sep|>`).
    *   Combine lines into the format: `input_phrase<|sep|>response_phrase<|eos|>`
    *   Save the result to `src/local_data/data/train.txt`.

## Training

Fine-tune the `distilgpt2` model on the prepared dataset:
```bash
python src/scripts/train.py
```
*   This script loads the pre-trained `distilgpt2` model and tokenizer.
*   It loads the `train.txt` dataset.
*   It fine-tunes the model using the `Trainer` API.
*   The fine-tuned model and tokenizer will be saved to `src/local_data/model/`.
*   Training uses CUDA (GPU) if available, otherwise CPU. Adjust `batch_size` in the script based on your GPU memory.

## Prediction

Generate responses using the fine-tuned model:

**Interactive Mode:**
```bash
python src/scripts/predict.py
```
Enter input phrases at the prompt. Type `quit` to exit.

**Batch Mode:**
```bash
python src/scripts/predict.py --input_file path/to/your_prompts.txt --output_file path/to/save_responses.txt
```
*   `--input_file`: Path to a file with one input prompt per line.
*   `--output_file`: (Optional) Path to save the generated responses. Defaults to `output_responses.txt` in the current directory.
*   Other generation parameters (like `--max_length`, `--num_beams`) can be adjusted via command-line arguments. See `python src/scripts/predict.py --help`.

## Project Structure

```
.
├── src/
│   ├── anime_chatbot/          # Core module definition (mainly setup.py)
│   │   ├── __init__.py
│   │   └── setup.py
│   ├── local_data/             # Data/model storage (gitignored by default)
│   │   ├── data/               # Processed datasets (input.txt, output.txt, train.txt)
│   │   └── model/              # Fine-tuned model/tokenizer
│   ├── scripts/                # Python scripts for data prep, training, prediction
│   │   ├── prep_data.py
│   │   ├── train.py
│   │   └── predict.py
│   └── tests/                  # Pytest tests
│       └── test_chatbot.py     # Placeholder for new tests
├── .gitignore
├── LICENSE
├── pytest.ini
└── README.md                   # This file
```

## Model Architecture

This project uses the `distilgpt2` model from Hugging Face, fine-tuned for the dialogue response generation task using a specific input format (`input<SEP>response<EOS>`).