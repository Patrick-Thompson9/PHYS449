# PHYS449 Assignment 3
- Variational Autoencoder for writing even numbers
- plots loss vs epochs
- Two hidden layers with RelU activation functions

## AI Statement
- Nearly all code was written using chatgpt, small features of functions written by myself
- Model structure and optimization done completely by chatgpt
- debugging done by chatgpt, copilot, and myself

## Dependencies
- json
- pandas
- torch
- torchvision
- os
- argparse
- matplotlib

## Running `main.py`

To run `main.py`, use

```sh
python main.py
```

You may edit the following run parameters by editing the run statement:
- epochs (int: number of epochs to train model)
- input_file (str: file path to data file)
- param_file (str: file path to parameter json file)
- n (int: number of sample images to create)
- verbose (True or false by simply including it or not: run program in verbose mode, i.e. print more)

replace the respctive values to run the program with your desired run parameters, edit this command

```sh
python main.py -epochs 10 -input_file 'data/even_mnist.csv' -param_file 'param/parameters.json' -n 100 --verbosity
```

There are more parameters to edit found in the parameters.json file in the param folder. These variables include:
- learning_rate: (float: step size of gradient descent)
- batch_size: (int: number of samples per batch)
- testing_size: (int: how much of the data will be set aside for testing)
- output_folder: (str: file path/name of the output folder)
