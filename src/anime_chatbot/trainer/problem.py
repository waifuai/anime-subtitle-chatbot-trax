"""
Generate a paraphrase for a given phrase using Trax.
"""

import os
import trax
from trax import layers as tl
from trax.supervised import training
from trax.data import inputs

# Define data generators for training and evaluation
def data_generator(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'r') as fout:
        for inp_line, tgt_line in zip(fin, fout):
            yield {'inputs': inp_line.strip(), 'targets': tgt_line.strip()}

# Hyperparameters for the Transformer model
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

# Hyperparameter ranges for tuning
def transformer_anime_chatbot_range(rhp):
    rhp = trax.RangedHParams(rhp)
    rhp.set_float("learning_rate", 0.05, 0.25, scale=rhp.LOG_SCALE)
    rhp.set_int("num_hidden_layers", 2, 8)
    rhp.set_discrete("d_model", [128, 256, 512])
    rhp.set_float("dropout", 0.4, 0.7)
    rhp.set_discrete("n_heads", [2, 4, 8, 16, 32, 64, 128])
    rhp.set_discrete("d_ff", [512, 1024])
    return rhp

# Create preprocessing pipeline
def create_input_pipeline(hparams, data_dir, train=True):
    # File paths
    input_file = os.path.join(data_dir, 'input.txt')
    output_file = os.path.join(data_dir, 'output.txt')
    
    # Tokenization
    tokenizer = trax.data.Tokenizer(vocab_size=hparams.vocab_size)
    
    # Processing pipeline
    pipeline = inputs.Serial(
        inputs.Tokenize(tokenizer=tokenizer),
        inputs.FilterByLength(max_length=2048),
        inputs.Shuffle(),
        inputs.BucketByLength(boundaries=[32, 256], batch_sizes=[16, 8, 4]),
        inputs.AddLossWeights()
    )
    
    return pipeline(data_generator(input_file, output_file))

# Create Transformer model
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

# Example usage
if __name__ == "__main__":
    # Initialize hyperparameters
    hparams = transformer_anime_chatbot()
    
    # Create data streams
    data_dir = 'data'
    train_stream = create_input_pipeline(hparams, data_dir)
    
    # Initialize model
    model = create_transformer_model(hparams)
    
    # Training task
    train_task = training.TrainTask(
        labeled_data=train_stream,
        loss_layer=tl.CrossEntropyLoss(),
        optimizer=trax.optimizers.Adam(hparams.learning_rate),
    )
    
    # Evaluation task (requires separate eval data)
    # eval_stream = create_input_pipeline(hparams, 'eval_data', train=False)
    # eval_task = training.EvalTask(...)
    
    # Training loop
    training_loop = training.Loop(
        model=model,
        tasks=train_task,
        output_dir='./training_output'
    )
    
    # Start training
    training_loop.run(n_steps=10000)