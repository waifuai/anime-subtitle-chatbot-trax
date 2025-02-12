import os
import trax
from trax.data import inputs

def data_generator(input_file, output_file):
    with open(input_file, 'r') as fin, open(output_file, 'r') as fout:
        for inp_line, tgt_line in zip(fin, fout):
            yield {'inputs': inp_line.strip(), 'targets': tgt_line.strip()}

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