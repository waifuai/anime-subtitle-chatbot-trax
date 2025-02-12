import unittest
from anime_chatbot.trainer import model, config
import trax

class TestModel(unittest.TestCase):
    def setUp(self):
        self.hparams = trax.HParams(
            vocab_size=2**10,
            d_model=32,
            d_ff=64,
            n_heads=2,
            dropout=0.1,
            num_hidden_layers=1
        )

    def test_create_transformer_model(self):
        transformer = model.create_transformer_model(self.hparams)
        self.assertIsNotNone(transformer)

if __name__ == '__main__':
    unittest.main()