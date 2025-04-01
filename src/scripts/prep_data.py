import os
from transformers import AutoTokenizer

def prepare_data(data_dir="src/local_data/data", model_name="distilgpt2"):
    """
    Reads input.txt and output.txt, combines them into the format
    'input<SEP>response<EOS>' suitable for fine-tuning a causal LM,
    and saves the result to train.txt.
    """
    input_file = os.path.join(data_dir, 'input.txt')
    output_file = os.path.join(data_dir, 'output.txt')
    train_file = os.path.join(data_dir, 'train.txt')

    # --- Tokenizer Setup ---
    # Define special tokens
    sep_token = "<|sep|>"
    # GPT2 uses <|endoftext|> as its EOS token by default
    
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add the separator token
    special_tokens_dict = {'sep_token': sep_token}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {num_added_toks} special token(s).")
    # Note: The model's embeddings will need resizing later during training setup

    # Ensure pad token is set (GPT2 usually doesn't have one by default)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # --- Data Processing ---
    print(f"Reading data from {input_file} and {output_file}")
    try:
        with open(input_file, 'r', encoding='utf-8') as fin, \
             open(output_file, 'r', encoding='utf-8') as fout, \
             open(train_file, 'w', encoding='utf-8') as ftrain:

            count = 0
            for inp_line, tgt_line in zip(fin, fout):
                inp_line = inp_line.strip()
                tgt_line = tgt_line.strip()

                if inp_line and tgt_line: # Ensure lines are not empty
                    # Format: input<SEP>response<EOS>
                    combined_line = f"{inp_line}{tokenizer.sep_token}{tgt_line}{tokenizer.eos_token}\n"
                    ftrain.write(combined_line)
                    count += 1
                else:
                    print(f"Skipping empty line pair: input='{inp_line}', target='{tgt_line}'")

            print(f"Successfully processed and wrote {count} lines to {train_file}")

    except FileNotFoundError as e:
        print(f"Error: Input or output file not found. {e}")
        print("Please ensure 'input.txt' and 'output.txt' exist in the data directory.")
    except Exception as e:
        print(f"An error occurred during data processing: {e}")

if __name__ == "__main__":
    # Example usage: Run this script directly to prepare the data
    # Assumes input.txt and output.txt are in src/local_data/data/
    prepare_data()