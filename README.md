# Background

From the original : https://keras.io/examples/nlp/lstm_seq2seq/#characterlevel-recurrent-sequencetosequence-model by François Chollet

*This example demonstrates how to implement a basic character-level recurrent sequence-to-sequence model. We apply it to translating short English sentences into short French sentences, character-by-character. Note that it is fairly unusual to do character-level machine translation, as word-level models are more common in this domain.*

A few extra things added are :

 - Wrapping classes : *Trainer , Predictor* and *Preprocessor* to better organize code
 - Two executables for operability : scripts/train.py and scripts/predict.py
 - Configurable set of parameters for I/O and training hyperparameters via the *config.txt* file.
 - Checkpointing, savingand loading of model weights

## Setup Instructions

1. Download this repository : git clone https://github.com/asierabreu/sequencer
2. Create a local python virtual environment :python3 -m venv virtualenv
3. Activate this environment : source virtualenv/bin/activate
4. Install dependencies : pip3 install -r requirements.txt
5. (Optional) Install kernel for virtual env: ipython kernel install --name "virtualenv" --user

## Folder structure

 - *training_files* : contains the input for the trainign process
 - *predict_input* : contains the input for the predict process
 - *predict_output* : contains the output of the predict process
 - *scripts* : contains Python executables
 - *models* : contains models from the training and checkpoints
 - *config.txt* : is the overal config file

## Configuration

The following configuration parameters are editable under the config.txt file:

 - train_dir=./training_files
 - model_dir=./models
 - predict_in=./predict_input
 - predict_out=./predict_output
 - latent_dim=256
 - num_samples=10000
 - optimizer=rmsprop
 - loss=categorical_crossentropy
 - metrics=accuracy
 - batch_size=64
 - epochs=100
 - val_split=0.2

## Usage 

 - Model training : python scripts/train.py
 - Model prediction : python scripts/predict.py