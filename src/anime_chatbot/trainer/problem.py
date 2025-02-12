"""
Generate a paraphrase for a given phrase using Trax.
"""

import trax
from trax import layers as tl
from trax.supervised import training
from anime_chatbot.trainer import data_utils, model, config

# Example usage
if __name__ == "__main__":
    # Initialize hyperparameters
    hparams = config.transformer_anime_chatbot()
    
    # Create data streams
    data_dir = 'data'
    train_stream = data_utils.create_input_pipeline(hparams, data_dir)
    
    # Initialize model
    model = model.create_transformer_model(hparams)
    
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