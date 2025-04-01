import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset

def train_model(
    data_dir="src/local_data/data",
    model_name="distilgpt2",
    output_dir="src/local_data/model",
    epochs=1, # Adjust as needed
    batch_size=4, # Adjust based on GPU memory
    learning_rate=5e-5,
):
    """
    Fine-tunes a causal LM (like distilgpt2) on the prepared train.txt dataset.
    """
    train_file = os.path.join(data_dir, 'train.txt')

    # --- Tokenizer and Model Setup ---
    sep_token = "<|sep|>"
    eos_token = "<|eos|>" # Define the new EOS token
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define and add special tokens (sep and new eos)
    # This ensures they are added before resizing embeddings
    special_tokens_dict = {'sep_token': sep_token, 'eos_token': eos_token}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    if num_added_toks > 0:
        print(f"Added {num_added_toks} special token(s): {special_tokens_dict}")

    # Set pad token if None
    # Set pad token to the new EOS token if pad_token is not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = eos_token # Use the new eos_token string
        print(f"Set pad_token to new eos_token: {tokenizer.pad_token}")

    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Resize embeddings to accommodate the new token
    model.resize_token_embeddings(len(tokenizer))
    print("Resized model token embeddings.")

    # --- Dataset Loading and Processing ---
    print(f"Loading dataset from {train_file}")
    try:
        # Load dataset directly from the text file
        raw_datasets = load_dataset('text', data_files={'train': train_file})
        print(f"Dataset loaded: {raw_datasets}")

        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize the text. The trainer expects 'input_ids'.
            # We don't need truncation/padding here as DataCollator handles it.
            return tokenizer(examples["text"])

        print("Tokenizing dataset...")
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"], # Remove original text column
            desc="Running tokenizer on dataset",
        )
        print(f"Dataset tokenized: {tokenized_datasets}")

    except FileNotFoundError:
        print(f"Error: Training file not found at {train_file}")
        print("Please run prep_data.py first.")
        return
    except Exception as e:
        print(f"An error occurred during dataset loading/processing: {e}")
        return

    # --- Training Setup ---
    # Data collator handles padding dynamically
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_steps=10_000, # Adjust saving frequency if needed
        save_total_limit=2, # Keep only the last 2 checkpoints
        logging_steps=100, # Log progress every 100 steps
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(), # Use mixed precision if CUDA is available
        # Add other arguments as needed (e.g., evaluation_strategy, weight_decay)
    )

    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        # eval_dataset=tokenized_datasets["validation"] # Add if you create a validation split
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Start Training ---
    print("Starting training...")
    try:
        train_result = trainer.train()
        print("Training finished.")

        # Save final model and tokenizer
        print(f"Saving model and tokenizer to {output_dir}")
        trainer.save_model() # Saves the model and tokenizer
        # Log metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("Model, metrics, and state saved.")

    except Exception as e:
        print(f"An error occurred during training: {e}")

if __name__ == "__main__":
    # Example usage: Run this script directly to start training
    # Assumes train.txt exists in src/local_data/data/
    train_model()