import trax
from trax import layers as tl

def create_transformer_model(hparams):
    return tl.Serial(
        tl.ShiftRight(),  # For autoregressive generation
        tl.Embedding(vocab_size=hparams.vocab_size, d_feature=hparams.d_model),
        tl.Transformer(
            d_model=hparams.d_model,
            d_ff=hparams.d_ff,
            n_heads=hparams.n_heads,
            dropout=hparams.dropout,
            n_encoder_layers=hparams.num_hidden_layers,
            n_decoder_layers=hparams.num_hidden_layers,
            mode='train'
        ),
        tl.Dense(hparams.vocab_size),
        tl.LogSoftmax()
    )