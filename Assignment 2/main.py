
import numpy as np
import os


def load_data(file_name='in.txt'):
    '''
    Load the input data using a file_name, assuming it is in the data folder.
    Return array of spins in form of integers.
    '''
    data = np.loadtxt('data/'+file_name, dtype=str)
    new_data = np.empty((len(data), len(data[0])))
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] == "+":
                new_data[i][j] = 1.0
            else:
                new_data[i][j] = -1.0
    return new_data

def energy(couplers, row):
    '''
    Calculates the energy of a row of spin sites using the couplers and spins.
    '''
    energy = 0
    for i in range(len(row)):
        # if on the last site, the neighbor will be the first site in the row
        if i == len(row) - 1:
            j = 0
        else:
            j = i + 1
        energy += row[i]*row[j]*couplers[i]
        return energy


def MCMC(couplers, current_state, new_state):
    '''
    Monte Carlo Markov Chain algorithm that moves from current_state to a new_state if 
    lower energy or by chance. Uses couplers to calculate energy.
    '''
    current_energy = energy(current_state)
    new_energy = energy(current_state)
    dE = new_energy - current_energy

    # if new state is lower energy, take it. If not, take it with some times
    if dE < 0:
        return new_state
    
    else:
        probability = np.exp(dE) # Beta = 1
        chance = random.uniform(0, 1)
        if chance < probability:
            return new_state
        else:
            return current_state

def make_random_row(N):
    '''
    make a random row of N spin sites (each 1 or -1)
    '''
    row = np.sign(np.random.random_sample((N,)) - 0.5)
    return row


def run_MCMC(couplers, trials=100, N=4):
    ''''
    Propagate down the Monte Carlo Markov Chain a number of trials. New random rows
    of N spins are made for each step.
    '''
    row1 = make_random_row(N)
    row2 = make_random_row(N)
    current_row = make_random_row(N)

    for i in range(trials):
        row2 = make_random_row(N)
        new_row = MCMC(couplers, current_row, row2)
        current_row = new_row

    return current_row


def main():
    load_data()
    


if __name__ == '__main__':
    main()
