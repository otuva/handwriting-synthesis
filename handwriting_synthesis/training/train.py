from handwriting_synthesis.config import processed_data_path, checkpoint_path, prediction_path
from handwriting_synthesis.rnn import RNN
from handwriting_synthesis.training import DataReader


def train():
    dr = DataReader(data_dir=processed_data_path)

    nn = RNN(
        reader=dr,
        log_dir='logs',
        checkpoint_dir=checkpoint_path,
        prediction_dir=prediction_path,
        learning_rates=[.0001, .00005, .00002],
        batch_sizes=[32, 64, 64],
        patiences=[1500, 1000, 500],
        beta1_decays=[.9, .9, .9],
        validation_batch_size=32,
        optimizer='rms',
        num_training_steps=100000,
        warm_start_init_step=0,
        regularization_constant=0.0,
        keep_prob=1.0,
        enable_parameter_averaging=False,
        min_steps_to_checkpoint=2000,
        log_interval=20,
        grad_clip=10,
        lstm_size=400,
        output_mixture_components=20,
        attention_mixture_components=10
    )
    nn.fit()
