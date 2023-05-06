import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import configparser
from preprocess import Preprocessor
import os
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Trainer:
    def __init__(self):
        # Read the configuration
        config = configparser.ConfigParser()
        config.read('./config.txt')
        self.train_dir = config.get('Files', 'train_dir')    # Absolute path of the top directory containing training data
        self.model_dir = config.get('Files', 'model_dir')    # Absolute path of the top directory to hold output model
        self.latent_dim = config.getint('Parameters', 'latent_dim')  # Latent dimensionality of the encoding space.
        self.num_samples = config.getint('Parameters', 'num_samples')# Number of samples to train on.
        self.optimizer = config.get('Training', 'optimizer')     # model optimizer algorithm
        self.loss = config.get('Training', 'loss')               # model loss function
        self.metrics = config.get('Training', 'metrics')         # model metrics (can be many, comma separated
        self.batch_size = config.getint('Training', 'batch_size')   # Batch size for training.
        self.epochs = config.getint('Training', 'epochs')           # Number of epochs to train for.
        self.val_split = config.getfloat('Training', 'val_split')     # Validation split

    def build(self,num_encoder_tokens,num_decoder_tokens):
        # model builder
        # Define an input sequence and process it.
        encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
        encoder = keras.layers.LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)

        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))

        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = keras.layers.LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = keras.layers.Dense(num_decoder_tokens, activation="softmax")
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        return model

    def train(self,model=None,encoder_input_data=None,decoder_input_data=None,decoder_target_data=None):
        # define callbacks for regular saving of weights at the end of every epoch
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.model_dir+os.sep+"checkpoints",
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # Model weights will be saved at the end of every epoch, if it's the best seen
        # so far.

        model.compile(
        optimizer=self.optimizer, loss=self.loss, metrics=self.metrics
        )
        # do the fitting
        model.fit(
            [encoder_input_data, decoder_input_data],
            decoder_target_data,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=self.val_split,
            callbacks=[model_checkpoint_callback]
        )
        # Save model (default keras format) at the end also
        model.save(self.model_dir+os.sep+"seq2seq")

    def run(self):
        # preprocess the input data
        preprocessing_outputs = Preprocessor(
            data_dir=self.train_dir,
            num_samples=self.num_samples
            ).run()
        
        for file,tokens in preprocessing_outputs.items():
            print('building model from : %s' %file)
            num_encoder_tokens=tokens['num_encoder_tokens']
            num_decoder_tokens=tokens['num_decoder_tokens']
            encoder_input_data=tokens['encoder_input_data']
            decoder_input_data=tokens['decoder_input_data']
            decoder_target_data=tokens['decoder_target_data']
            # build the model from scratch
            model = self.build(
                num_encoder_tokens=num_encoder_tokens,
                num_decoder_tokens=num_decoder_tokens
                )
            # train it
            self.train(
                model=model,
                encoder_input_data=encoder_input_data,
                decoder_input_data=decoder_input_data,
                decoder_target_data=decoder_target_data
                )

# Execution   
if __name__ == "__main__":
    print("------------------")
    print("Start training ...")
    print("------------------")
    Trainer().run()


