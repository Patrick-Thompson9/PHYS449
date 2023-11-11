# PHYS449
- Boltzmann machine to predict Ising model coupler values

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

Or 

```sh
python main.py data/in.txt --epochs 50 --learning_rate 0.001 --output_folder 'output'
```
Where you enter your desired values for each hyperparameter.