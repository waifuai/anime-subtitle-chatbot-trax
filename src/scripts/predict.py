import torch
import argparse
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Moved generate_response outside the main predict function for testability
def generate_response(
    prompt_text,
    model,
    tokenizer,
    device,
    sep_token,
    max_length=100,
    num_beams=5,
    no_repeat_ngram_size=2,
    early_stopping=True,
):
    """Generates a response for a given prompt using the model."""
    # Format input: "input<SEP>"
    formatted_prompt = f"{prompt_text}{sep_token}"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    # Generate output sequence
    output_sequences = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'], # Include attention mask
        max_length=inputs['input_ids'].shape[1] + max_length, # Max length relative to input
        num_beams=num_beams,
        no_repeat_ngram_size=no_repeat_ngram_size,
        early_stopping=early_stopping,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # Decode the generated sequence, skipping special tokens and the prompt
    generated_text = tokenizer.decode(
        output_sequences[0], # Get the first sequence from the batch
        skip_special_tokens=False # Keep special tokens initially for splitting
    )

    # Extract only the response part after the separator
    # Find the separator token in the generated text
    parts = generated_text.split(sep_token)
    if len(parts) > 1:
        # Get the part after the separator and remove the EOS token
        response = parts[1].replace(tokenizer.eos_token, "").strip()
        return response
    else:
        # If separator not found (shouldn't happen ideally), return decoded output minus prompt
        prompt_tokens_len = inputs['input_ids'].shape[1]
        response_tokens = output_sequences[0][prompt_tokens_len:]
        response = tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        # Add a fallback log or warning here if needed
        print(f"Warning: Separator token '{sep_token}' not found in generated text: {generated_text}")
        return response


def predict(
    model_dir="src/local_data/model",
    input_file=None,
    output_file=None,
    max_length=100, # Max length for generated response
    num_beams=5,    # Beam search parameters
    no_repeat_ngram_size=2,
    early_stopping=True,
):
    """
    Loads the model and tokenizer, then generates responses.
    Supports interactive mode or batch processing from a file.
    """
    print(f"Loading model and tokenizer from {model_dir}...")
    try:
        # Load with trust_remote_code=True if necessary for custom models/tokenizers
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
    except OSError:
        print(f"Error: Model or tokenizer not found in {model_dir}.")
        print("Please ensure the model has been trained and saved.")
        return
    except Exception as e:
        print(f"An error occurred loading the model/tokenizer: {e}")
        return

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Using device: {device}")

    # Ensure pad token is set for generation
    # Ensure pad token is set (use the EOS token if not specified)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
             # Fallback if EOS is somehow also None (shouldn't happen with saved tokenizer)
             print("Warning: Tokenizer has no pad_token and no eos_token. Generation might fail.")
             # Handle this case appropriately, maybe raise an error or set a default?
             # For now, let's try setting a default, though this indicates a problem.
             tokenizer.add_special_tokens({'pad_token': '<|pad|>'}) # Add a default pad token
             print("Added default pad token '<|pad|>'.")
        else:
             tokenizer.pad_token = tokenizer.eos_token
             print(f"Set pad_token to eos_token: {tokenizer.pad_token}")


    # Determine the separator token - prioritize tokenizer's definition
    sep_token = getattr(tokenizer, 'sep_token', '<|sep|>') # Use getattr for safer access
    if not sep_token: # Handle cases where sep_token might be None or empty
        sep_token = "<|sep|>"
        print(f"Warning: Tokenizer's sep_token is missing or empty. Using default: {sep_token}")


    # --- Mode Selection ---
    if input_file:
        # Batch mode
        print(f"Batch mode: Reading from {input_file}, writing to {output_file or 'output_responses.txt'}")
        output_filename = output_file or "output_responses.txt"
        try:
            with open(input_file, 'r', encoding='utf-8') as fin, \
                 open(output_filename, 'w', encoding='utf-8') as fout:
                count = 0
                for line in fin:
                    prompt = line.strip()
                    if prompt:
                        # Call the refactored generate_response function
                        response = generate_response(
                            prompt_text=prompt,
                            model=model,
                            tokenizer=tokenizer,
                            device=device,
                            sep_token=sep_token,
                            max_length=max_length,
                            num_beams=num_beams,
                            no_repeat_ngram_size=no_repeat_ngram_size,
                            early_stopping=early_stopping,
                        )
                        fout.write(response + '\n')
                        count += 1
                        if count % 10 == 0:
                            print(f"Processed {count} lines...")
                print(f"Finished processing {count} lines. Output saved to {output_filename}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {input_file}")
        except Exception as e:
            print(f"An error occurred during batch processing: {e}")
    else:
        # Interactive mode
        print("Interactive mode. Enter 'quit' to exit.")
        while True:
            prompt = input("Input: ")
            if prompt.lower() == 'quit':
                break
            if prompt:
                # Call the refactored generate_response function
                response = generate_response(
                    prompt_text=prompt,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    sep_token=sep_token,
                    max_length=max_length,
                    num_beams=num_beams,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    early_stopping=early_stopping,
                )
                print(f"Response: {response}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate responses using the fine-tuned chatbot model.")
    parser.add_argument("--model_dir", type=str, default="src/local_data/model", help="Directory containing the fine-tuned model and tokenizer.")
    parser.add_argument("--input_file", type=str, default=None, help="Path to a file containing input prompts (one per line) for batch mode.")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save the generated responses in batch mode (defaults to output_responses.txt).")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum number of tokens to generate for the response.")
    parser.add_argument("--num_beams", type=int, default=5, help="Number of beams for beam search.")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=2, help="Size of n-grams that cannot be repeated.")
    parser.add_argument("--early_stopping", action=argparse.BooleanOptionalAction, default=True, help="Enable/disable early stopping in beam search.")


    args = parser.parse_args()
    predict(
        model_dir=args.model_dir,
        input_file=args.input_file,
        output_file=args.output_file,
        max_length=args.max_length,
        num_beams=args.num_beams,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        early_stopping=args.early_stopping,
    )