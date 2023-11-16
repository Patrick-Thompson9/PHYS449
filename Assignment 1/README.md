# PHYS449 Assignment 1
- Recurent Neural Network for binary multiplication
- RelU activation function
- Sigmoid output layer
- Plots loss, saves plot and model in Ouput Folder

## AI Statement
- All code was generated through many chatgpt prompts
- Model structure was optimized through my knowledge as well as chatgpt suggestions
  
## Dependencies

- torch
- argparse
- matplotlib
- numpy
- os

## Running `main.py`

To run `main.py`, use

```sh
python main.py 
```

Or edit the following variables by entering your desired value instead of what is in the cope snippet:
- epochs: number of epochs for model training.
- batch_size: batch size for model training.
- random_seed: random number to allow reproducibility.
- learning_rate: Step size for gradient descent.
- training_size: desired size of training data set.
- test_size: desired size of test data set.

```sh
python main.py --epochs 50 --batch_size 32 --random_seed 42 --learning_rate 0.0001 --training_size 8000 --test_size 2000
```
