import pytest
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

# Import functions from the scripts
from src.scripts import prep_data
from src.scripts.predict import generate_response # Import the refactored function
from transformers import AutoTokenizer # Needed for prep_data test

# --- Test Data Preparation ---

def test_data_preparation(tmp_path):
    """Tests the prep_data.prepare_data function."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"
    train_file = data_dir / "train.txt"

    # Create dummy input/output files
    input_content = "Hello there.\nHow are you?\n"
    output_content = "General Kenobi.\nI am fine.\n"
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    # Define expected tokens (use a real tokenizer instance for accuracy)
    # Using distilgpt2 as specified in the plan
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    sep_token = "<|sep|>"
    eos_token = "<|eos|>" # Use the new EOS token
    # Add both special tokens for the test tokenizer instance
    tokenizer.add_special_tokens({'sep_token': sep_token, 'eos_token': eos_token})

    # Run the preparation function
    prep_data.prepare_data(data_dir=str(data_dir), model_name=model_name)

    # Assertions
    assert train_file.exists()
    content = train_file.read_text(encoding='utf-8').strip().splitlines()
    assert len(content) == 2
    expected_line1 = f"Hello there.{sep_token}General Kenobi.{eos_token}"
    expected_line2 = f"How are you?{sep_token}I am fine.{eos_token}"
    assert content[0] == expected_line1
    assert content[1] == expected_line2

def test_data_preparation_file_not_found(tmp_path, capsys):
    """Tests prep_data handling of missing input files."""
    data_dir = tmp_path / "data"
    # Don't create the directory or files
    prep_data.prepare_data(data_dir=str(data_dir))
    captured = capsys.readouterr()
    assert "Error: Input or output file not found" in captured.out or "Error: Input or output file not found" in captured.err

# --- Test Prediction Logic ---

@pytest.fixture
def mock_tokenizer():
    """Fixture for a mocked tokenizer."""
    tokenizer = MagicMock()
    tokenizer.sep_token = "<|sep|>"
    tokenizer.eos_token = "<|eos|>" # Use the new EOS token
    # Assuming the new token gets a new ID, let's use a placeholder ID for the mock
    # In reality, this ID would be determined when adding the token.
    # Let's assume it's 50257 for the mock. Pad ID might also change.
    tokenizer.pad_token_id = 50257 # Mock ID for new EOS/PAD
    tokenizer.eos_token_id = 50257 # Mock ID for new EOS
    # Mock the __call__ method to return an object with a .to() method
    mock_tokenized_output = MagicMock()
    mock_tensor_dict = {
        'input_ids': MagicMock(shape=[1, 5]), # Mock tensor shape
        'attention_mask': MagicMock()
    }
    # Configure the .to() method on the mock output object
    mock_tokenized_output.to.return_value = mock_tensor_dict
    # Make the tokenizer call return this mock object
    tokenizer.return_value = mock_tokenized_output
    # Mock the decode method
    def mock_decode(ids, skip_special_tokens=False):
        # Simulate decoding based on whether special tokens are skipped
        if skip_special_tokens:
            return "Generated response text"
        else:
            # Simulate raw output including prompt, sep, response, eos
            return f"Input prompt{tokenizer.sep_token}Generated response text{tokenizer.eos_token}" # Uses the updated tokenizer.eos_token
    tokenizer.decode = mock_decode
    return tokenizer

@pytest.fixture
def mock_model():
    """Fixture for a mocked model."""
    model = MagicMock()
    # Mock the generate method to return some dummy output sequence
    # The exact tensor content doesn't matter much here, just the structure
    mock_output_tensor = MagicMock()
    # Simulate slicing behavior for the fallback case in generate_response
    mock_output_tensor.__getitem__.return_value = [1, 2, 3] # Dummy token IDs
    model.generate.return_value = [mock_output_tensor] # Return a list containing the tensor
    return model

@pytest.fixture
def mock_device():
    """Fixture for a mocked device (CPU)."""
    return "cpu" # Simulate running on CPU for tests

def test_generate_response_normal(mock_model, mock_tokenizer, mock_device):
    """Tests the generate_response function with mocked components."""
    prompt = "Input prompt"
    sep_token = mock_tokenizer.sep_token

    response = generate_response(
        prompt_text=prompt,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=mock_device,
        sep_token=sep_token,
        max_length=50 # Example value
    )

    # Assertions
    mock_tokenizer.assert_called_with(f"{prompt}{sep_token}", return_tensors="pt")
    mock_model.generate.assert_called_once()
    # Check if the decoded response is extracted correctly
    assert response == "Generated response text"

def test_generate_response_separator_fallback(mock_model, mock_tokenizer, mock_device, capsys):
    """Tests the fallback logic when separator is not in the decoded output."""
    prompt = "Input prompt"
    sep_token = mock_tokenizer.sep_token

    # Modify the mock decode to *not* include the separator
    mock_tokenizer.decode = MagicMock(return_value=f"Input prompt Generated response text{mock_tokenizer.eos_token}") # Uses updated mock_tokenizer.eos_token

    response = generate_response(
        prompt_text=prompt,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=mock_device,
        sep_token=sep_token,
        max_length=50
    )

    # Assertions
    captured = capsys.readouterr()
    assert f"Warning: Separator token '{sep_token}' not found" in captured.out
    # The fallback uses decode with skip_special_tokens=True
    mock_tokenizer.decode.assert_called_with([1, 2, 3], skip_special_tokens=True)
    # Check if the fallback response is returned (based on the modified mock decode)
    # This assertion depends on how the fallback decode is mocked.
    # If the fallback decode (skip_special_tokens=True) returns "Generated response text", this works.
    # Let's adjust the mock decode for the fallback scenario:
    def complex_mock_decode(ids, skip_special_tokens=False):
         if skip_special_tokens:
             return "Fallback response" # Specific text for fallback
         else:
             return f"Input prompt Generated response text{mock_tokenizer.eos_token}" # Output without separator, uses updated mock_tokenizer.eos_token
    mock_tokenizer.decode = complex_mock_decode

    # Re-run with the adjusted mock
    response = generate_response(
        prompt_text=prompt,
        model=mock_model,
        tokenizer=mock_tokenizer,
        device=mock_device,
        sep_token=sep_token,
        max_length=50
    )
    assert response == "Fallback response"


# Placeholder for potential future tests (e.g., testing the main predict() CLI interaction)
# def test_predict_cli_batch_mode(tmp_path):
#     pass

# def test_predict_cli_interactive_mode():
#     pass