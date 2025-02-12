import unittest
from anime_chatbot.trainer import data_utils
import os
import trax

class TestDataUtils(unittest.TestCase):
    def setUp(self):
        # Create dummy input and output files for testing
        self.input_file = "test_input.txt"
        self.output_file = "test_output.txt"
        with open(self.input_file, "w") as f:
            f.write("input sentence 1\ninput sentence 2")
        with open(self.output_file, "w") as f:
            f.write("output sentence 1\noutput sentence 2")
        self.hparams = trax.HParams(vocab_size=2**10)  # Dummy hparams
        self.data_dir = "."  # Current directory

    def tearDown(self):
        # Remove dummy files
        os.remove(self.input_file)
        os.remove(self.output_file)

    def test_data_generator(self):
        generator = data_utils.data_generator(self.input_file, self.output_file)
        data = list(generator)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]['inputs'], "input sentence 1")
        self.assertEqual(data[0]['targets'], "output sentence 1")

    def test_create_input_pipeline(self):
        pipeline = data_utils.create_input_pipeline(self.hparams, self.data_dir)
        # Basic check to see if the pipeline returns something
        self.assertIsNotNone(pipeline)

if __name__ == '__main__':
    unittest.main()