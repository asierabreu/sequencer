# Run the inference (sampling)
import os
import numpy as np
import tensorflow as tf
import configparser
from tensorflow import keras
from datetime import datetime
from preprocess import Preprocessor
# disable tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

class Predictor():
    # 1. encode input and retrieve initial decoder state
    # 2. run one step of decoder with this initial state and a "start of sequence" token as target. Output will be the next target token.
    # 3. Repeat with the current target token and current states
    def __init__(self):
        # Read the configuration
        config = configparser.ConfigParser()
        config.read('./config.txt')
        self.predict_in = config.get('Files', 'predict_in')    # Absolute path of the top directory containing input data
        self.predict_out = config.get('Files', 'predict_out')    # Absolute path of the top directory containing output
        self.model_dir = config.get('Files', 'model_dir')    # Absolute path of the top directory to hold output model
        self.latent_dim = config.getint('Parameters', 'latent_dim')  # Latent dimensionality of the encoding space.
        self.num_samples = config.getint('Parameters', 'num_samples')# Number of samples to train on.
        
    def load(self):

        # Restore the pre-trained model
        model = keras.models.load_model(self.model_dir+os.sep+"seq2seq")

        # construct the encoder and decoder
        encoder_inputs = model.input[0]  # input_1
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output  # lstm_1
        encoder_states = [state_h_enc, state_c_enc]
        encoder_model = keras.Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1]  # input_2
        decoder_state_input_h = keras.Input(shape=(self.latent_dim,))
        decoder_state_input_c = keras.Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs
        )
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = keras.Model(
            [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
        )
        return encoder_model,decoder_model

    def reverse_lookup(self,input_token_index,target_token_index):
        # Reverse-lookup token index to decode sequences back to something readable.
        reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
        reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())
        return reverse_input_char_index,reverse_target_char_index

    def decoder(self,
                input_seq,
                encoder_model,
                decoder_model,
                reverse_target_char_index,
                num_decoder_tokens,
                target_token_index,
                max_decoder_seq_length):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index["\t"]] = 1.0

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ""
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if sampled_char == "\n" or len(decoded_sentence) > max_decoder_seq_length:
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.0

            # Update states
            states_value = [h, c]
        return decoded_sentence
    
    def predict(self,input_texts,encoder_input_data,encoder_model,decoder_model,reverse_target_char_index,num_decoder_tokens,target_token_index,sequence_length=20):
        # write decoded sequence
        now= datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        with open(self.predict_out+os.sep+"decoded_sentences_"+now+".txt", "w") as write_file:
            for seq_index in range(sequence_length):
                # Take one sequence (part of the training set) for trying out decoding.
                input_seq = encoder_input_data[seq_index : seq_index + 1]
                input_sentence = input_texts[seq_index]
                decoded_sentence = self.decoder(
                    input_seq=input_seq,
                    encoder_model=encoder_model,
                    decoder_model=decoder_model,
                    reverse_target_char_index=reverse_target_char_index,
                    num_decoder_tokens=num_decoder_tokens,
                    target_token_index=target_token_index,
                    max_decoder_seq_length=sequence_length
                    )
                print("-")
                print(" Input sentence :", input_texts[seq_index])
                print("Decoded sentence:", decoded_sentence)
                # write decoded sentence
                write_file.write('   sequence index : %2d' %seq_index)
                write_file.write('   input sentence : %s' %input_sentence)
                write_file.write(' decoded sentence : %s' %decoded_sentence)

    def run(self):

        # preprocess the input data
        preprocessing_outputs = Preprocessor(
            data_dir=self.predict_in,
            num_samples=self.num_samples
            ).run()
    
        # Load the pre-trained model (encoder+decoder)
        encoder_model,decoder_model = self.load()

        for file,tokens in preprocessing_outputs.items():
            print('file preprocessed : %s' %file)
            input_texts=tokens['input_texts']
            num_encoder_tokens=tokens['num_encoder_tokens']
            num_decoder_tokens=tokens['num_decoder_tokens']
            encoder_input_data=tokens['encoder_input_data']
            decoder_input_data=tokens['decoder_input_data']
            decoder_target_data=tokens['decoder_target_data']
            input_token_index=tokens['input_token_index']
            target_token_index=tokens['target_token_index']
            
            # reverse look up on input
            reverse_input_char_index,reverse_target_char_index = self.reverse_lookup(
                input_token_index=input_token_index,
                target_token_index=target_token_index
                )

            # make the prediction 
            self.predict(input_texts=input_texts,
                        encoder_input_data=encoder_input_data,
                        encoder_model=encoder_model,
                        decoder_model=decoder_model,
                        reverse_target_char_index=reverse_target_char_index,
                        num_decoder_tokens=num_decoder_tokens,
                        target_token_index=target_token_index
                        )
        
# Execution    
if __name__ == "__main__":
    print("--------------------")
    print("doing prediction ...")
    print("--------------------")
    Predictor().run()
