## Setup Instructions

The following instructions assume you have downloaded the git repository , e.g. by doing: 

git clone https://github.com/asierabreu/sequencer

1. Create a python virtual environment :python3 -m venv virtualenv
2. Activate this environment : source virtualenv/bin/activate
3. Install dependencies : pip3 install -r requirements.txt
4. (Optional) Install kernel for virtual env: ipython kernel install --name "virtualenv" --user

## Folder structure

 - training_files : contains the input for the trainign process
 - predict_input : contains the input for the predict process
 - predict_input : contains the output of the predict process
 - scripts : contains Python executables
 - models : contains output models saved during training
 - config.txt : is the overal config file

## Usage 

Model training : python scripts/train.py
Model prediction : python scripts/predict.py