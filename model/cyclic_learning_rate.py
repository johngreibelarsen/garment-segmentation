""" 
    Helper function to illustrate the shape and form of our bespoke learning rate function 
"""

import numpy as np
import matplotlib.pyplot as plt  

import util.params as params

plateau_steps = params.plateau_steps
step_size = params.step_size 
min_lr = params.min_lr
max_lr = params.max_lr


def plateau_then_triangular_lr(epochs):    
    """
        Given the inputs, calculates the lr that should be applicable for this epoch.
        The algo will start out with a LR equal to the average of the min and max rates
        for the first no. of plateau_steps. Hereafter the algo will start oscillating 
        with a wave length around 2xstep_size
    """

    if epochs < plateau_steps:
        return (min_lr + max_lr)/2
    
    cycle = np.floor(1+epochs/(2*step_size))
    x = np.abs(epochs/step_size - 2*cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1-x))
    return lr



def plateau_then_decl_triangular_lr(epochs):    
    """
        Given the inputs, calculates the lr that should be applicable for this epoch.
        The algo will start out with a LR equal to the average of the min and max rates
        for the first no. of plateau_steps. Hereafter the algo will start oscillating 
        with a wave length around 2xstep_size and an ever declining amplitude (the minimal
        LR is unaffected however the max LR will keep declining in an power based fashion) 
    """

    if epochs < plateau_steps:
        return (min_lr + max_lr)/2
    
    cycle = np.floor(1+epochs/(2*step_size))
    x = np.abs(epochs/step_size - 2*cycle + 1)
    lr = min_lr + (max_lr - min_lr) * np.maximum(0, (1-x))/float(1.2**(cycle-1))
    return lr


def show_function(lr_function):
    """ Helper function to illustrate the shape and form of our bespoke learning rate function """
    epochs = 100
    lr_trend = list()
    for epoch in range(epochs):
        lr = lr_function(epoch)
        print(lr)
        lr_trend.append(lr)
    
    plt.plot(lr_trend, marker='o', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    
    
    
# Demo of how the LR varies with epocs
if __name__ == '__main__':
    show_function(plateau_then_decl_triangular_lr)
