import numpy as np
import glob
import os

class Preprocessor():
    
    def __init__(self,data_dir,num_samples): 
        self.data_dir = data_dir       # Absolute path of the top directory containing data
        self.num_samples = num_samples # Number of samples to train on.

    def vectorize(self,input_file=None):
        # Vectorize the data , do the embedding
        input_texts = []
        target_texts = []
        input_characters = set()
        target_characters = set()
        #print("opening input file : %s" %input_file)
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.read().split("\n")
        for line in lines[: min(self.num_samples, len(lines) - 1)]:
            input_text, target_text, _ = line.split("\t")
            # We use "tab" as the "start sequence" character
            # for the targets, and "\n" as "end sequence" character.
            target_text = "\t" + target_text + "\n"
            input_texts.append(input_text)
            target_texts.append(target_text)
            for char in input_text:
                if char not in input_characters:
                    input_characters.add(char)
            for char in target_text:
                if char not in target_characters:
                    target_characters.add(char)

        input_characters = sorted(list(input_characters))
        target_characters = sorted(list(target_characters))
        num_encoder_tokens = len(input_characters)
        num_decoder_tokens = len(target_characters)
        max_encoder_seq_length = max([len(txt) for txt in input_texts])
        max_decoder_seq_length = max([len(txt) for txt in target_texts])
        print("")
        print("              Number of samples:", len(input_texts))
        print("Number of unique input tokens  :", num_encoder_tokens)
        print("Number of unique output tokens :", num_decoder_tokens)
        print("Max sequence length for inputs :", max_encoder_seq_length)
        print("Max sequence length for outputs:", max_decoder_seq_length)

        input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        encoder_input_data = np.zeros(
            (len(input_texts), max_encoder_seq_length, num_encoder_tokens), dtype="float32"
        )
        decoder_input_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )
        decoder_target_data = np.zeros(
            (len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32"
        )

        for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, input_token_index[char]] = 1.0
            encoder_input_data[i, t + 1 :, input_token_index[" "]] = 1.0
            for t, char in enumerate(target_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                decoder_input_data[i, t, target_token_index[char]] = 1.0
                if t > 0:
                    # decoder_target_data will be ahead by one timestep
                    # and will not include the start character.
                    decoder_target_data[i, t - 1, target_token_index[char]] = 1.0
            decoder_input_data[i, t + 1 :, target_token_index[" "]] = 1.0
            decoder_target_data[i, t:, target_token_index[" "]] = 1.0
        # create a dictionary with outputs and other required items
        tokens={}
        tokens['num_encoder_tokens']=num_encoder_tokens
        tokens['num_decoder_tokens']=num_decoder_tokens
        tokens['input_token_index']=input_token_index
        tokens['target_token_index']=target_token_index
        tokens['encoder_input_data']=encoder_input_data
        tokens['decoder_input_data']=decoder_input_data
        tokens['decoder_target_data']=decoder_target_data
        return tokens

    def run(self):
        # vectorize each file in the input directory
        # create a dictonary, with N entries one for each file inside the inptu directory 
        # each entry contains the outputs from the vectorizer for each file 
        preprocessing_outputs={}
        input_files=glob.glob(self.data_dir+os.sep+"*.txt")
        print('%2d input files found under %s' %(len(input_files),self.data_dir))
        for file in input_files:
            print('preprocessing input file : %s' %file)
            preprocessing_outputs[file]=self.vectorize(file)
        return preprocessing_outputs