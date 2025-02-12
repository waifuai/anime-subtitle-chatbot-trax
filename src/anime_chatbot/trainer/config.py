import trax

def transformer_anime_chatbot():
    return trax.HParams(
        num_hidden_layers=2,
        d_model=128,
        d_ff=512,
        n_heads=4,
        dropout=0.6,
        learning_rate=0.05,
        vocab_size=2**16  # ~64k
    )

def transformer_anime_chatbot_range(rhp):
    rhp = trax.RangedHParams(rhp)
    rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
    rhp.set_int("num_hidden_layers", 2, 8)
    rhp.set_discrete("d_model", [128, 256, 512])
    rhp.set_float("dropout", 0.4, 0.7)
    rhp.set_discrete("n_heads", [2, 4, 8, 16, 32, 64, 128])
    rhp.set_discrete("d_ff", [512, 1024])
    return rhp