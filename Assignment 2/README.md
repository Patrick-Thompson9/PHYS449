# PHYS449 Assignment 2
- Boltzmann machine to predict Ising model coupler values
- Plots loss vs epochs
- two hidden layers with RelU activation functions

## AI Statement
- All code was generated through many chatgpt prompts
- Model structure was optimized through my knowledge as well as chatgpt suggestions

## Dependencies

- numpy
- torch
- matplotlib
- os
- argparse

## Running `main.py`

To run `main.py`, use

```sh
python main.py data/in.txt
```

You may edit the following run parameters by editing the run statement:
- file_path (str: file path to data file)
- epochs (int: number of epochs to train model)
- learning_rate (float: step size/learning rate for SGD)
- output_folder (str: file path/name of output folder)
- verbose (True or false by simply including it or not: run program in verbose mode, i.e. print more)

replace the respctive values to run the program with your desired run parameters, edit this command

```sh
python main.py data/in.txt --epochs 50 --learning_rate 0.001 --output_folder 'output' --verbose
```
Where you enter your desired values for each hyperparameter.