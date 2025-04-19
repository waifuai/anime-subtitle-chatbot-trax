import pytest
import os
import pathlib
from unittest.mock import MagicMock, patch, mock_open

# Import functions from the updated predict script
from src.scripts.predict import (
    load_api_key,
    load_examples,
    generate_gemini_response,
    predict, # Import main function for potential CLI tests later
    API_KEY_FILENAME,
    MODEL_NAME
)

# --- Test Helper Functions ---

def test_load_examples_success(tmp_path):
    """Tests loading examples successfully."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"

    input_content = "Hello?\nHow are you?\n"
    output_content = "Hi!\nI'm good.\n"
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    examples = load_examples(str(input_file), str(output_file))

    assert len(examples) == 2
    assert examples[0] == ("Hello?", "Hi!")
    assert examples[1] == ("How are you?", "I'm good.")

def test_load_examples_file_not_found(tmp_path, capsys):
    """Tests loading examples when files are missing."""
    input_file = tmp_path / "nonexistent_input.txt"
    output_file = tmp_path / "nonexistent_output.txt"

    examples = load_examples(str(input_file), str(output_file))
    captured = capsys.readouterr()

    assert examples == []
    assert "Error: Example file not found" in captured.out

def test_load_examples_empty_lines(tmp_path):
    """Tests that empty lines are skipped when loading examples."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    input_file = data_dir / "input.txt"
    output_file = data_dir / "output.txt"

    input_content = "Line 1\n\nLine 3"
    output_content = "Resp 1\nResp 2\nResp 3" # Mismatched lines intentionally
    input_file.write_text(input_content, encoding='utf-8')
    output_file.write_text(output_content, encoding='utf-8')

    examples = load_examples(str(input_file), str(output_file))

    # Only pairs with non-empty input and output are kept
    assert len(examples) == 2
    assert examples[0] == ("Line 1", "Resp 1")
    assert examples[1] == ("Line 3", "Resp 3")


def test_load_api_key_success(tmp_path):
    """Tests loading the API key successfully."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    api_key_file = home_dir / API_KEY_FILENAME
    api_key_content = "test_api_key_123"
    api_key_file.write_text(api_key_content, encoding='utf-8')

    with patch('pathlib.Path.home', return_value=home_dir):
        api_key = load_api_key(api_key_file) # Pass the specific path for test isolation
        assert api_key == api_key_content

def test_load_api_key_file_not_found(tmp_path, capsys):
    """Tests loading API key when the file is missing."""
    home_dir = tmp_path / "home"
    home_dir.mkdir()
    api_key_file = home_dir / API_KEY_FILENAME # Path exists, but file doesn't

    with patch('pathlib.Path.home', return_value=home_dir):
        api_key = load_api_key(api_key_file)
        captured = capsys.readouterr()
        assert api_key is None
        assert f"Error: API key file not found at {api_key_file}" in captured.out

# --- Test Gemini API Interaction (using mocks) ---

@pytest.fixture
def mock_gemini_client():
    """Fixture for a mocked google.generativeai client and model."""
    with patch('google.generativeai.configure') as mock_configure, \
         patch('google.generativeai.GenerativeModel') as mock_GenerativeModel:

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        # Simulate a successful response with text
        mock_response.parts = [MagicMock()] # Simulate having parts
        mock_response.text = "Generated anime response."
        mock_response.prompt_feedback = None # No blocking
        mock_model_instance.generate_content.return_value = mock_response

        mock_GenerativeModel.return_value = mock_model_instance

        yield {
            "configure": mock_configure,
            "GenerativeModel": mock_GenerativeModel,
            "model_instance": mock_model_instance,
            "mock_response": mock_response
        }

def test_generate_gemini_response_success(mock_gemini_client):
    """Tests successful response generation using mocked Gemini API."""
    api_key = "fake_key"
    prompt = "User input prompt"
    examples = [("Ex Input 1", "Ex Output 1"), ("Ex Input 2", "Ex Output 2")]

    response_text = generate_gemini_response(api_key, prompt, examples)

    # Assertions
    mock_gemini_client["configure"].assert_called_once_with(api_key=api_key)
    mock_gemini_client["GenerativeModel"].assert_called_once_with(MODEL_NAME)
    mock_gemini_client["model_instance"].generate_content.assert_called_once()

    # Check prompt construction (basic check)
    call_args, _ = mock_gemini_client["model_instance"].generate_content.call_args
    generated_prompt = call_args[0]
    assert "You are an anime chatbot." in generated_prompt
    assert "Input: Ex Input 1" in generated_prompt
    assert "Output: Ex Output 1" in generated_prompt
    assert "Input: Ex Input 2" in generated_prompt
    assert "Output: Ex Output 2" in generated_prompt
    assert f"Input: {prompt}" in generated_prompt
    assert generated_prompt.endswith("Output:")

    assert response_text == "Generated anime response."

def test_generate_gemini_response_blocked(mock_gemini_client, capsys):
    """Tests handling of a blocked prompt response."""
    api_key = "fake_key"
    prompt = "Problematic prompt"
    examples = []

    # Configure mock response for blocking
    mock_gemini_client["mock_response"].parts = [] # No parts when blocked
    mock_gemini_client["mock_response"].text = None # No text when blocked
    mock_gemini_client["mock_response"].prompt_feedback = MagicMock()
    mock_gemini_client["mock_response"].prompt_feedback.block_reason = "SAFETY"
    mock_gemini_client["model_instance"].generate_content.return_value = mock_gemini_client["mock_response"]


    response_text = generate_gemini_response(api_key, prompt, examples)
    captured = capsys.readouterr()

    assert "Warning: Prompt blocked. Reason: SAFETY" in captured.out
    assert response_text == "[Blocked: SAFETY]"

def test_generate_gemini_response_api_error(mock_gemini_client, capsys):
    """Tests handling of an exception during API call."""
    api_key = "fake_key"
    prompt = "User input"
    examples = []

    # Configure mock to raise an exception
    mock_gemini_client["model_instance"].generate_content.side_effect = Exception("API connection failed")

    response_text = generate_gemini_response(api_key, prompt, examples)
    captured = capsys.readouterr()

    assert response_text is None
    assert "An error occurred calling the Gemini API: API connection failed" in captured.out

# --- Placeholder for potential future CLI tests ---
# def test_predict_cli_batch_mode(tmp_path, mock_gemini_client):
#     # Needs more setup: mock file I/O, mock Path.home, mock argparse
#     pass

# def test_predict_cli_interactive_mode(mock_gemini_client):
#     # Needs more setup: mock input(), mock Path.home
#     pass