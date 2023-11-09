
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


def main():
    load_data()


if __name__ == '__main__':
    main()
