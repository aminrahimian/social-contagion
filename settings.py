# Params and other settings are set here
# Settings are for the generative model as well as the inference engine


# The generative model settings

import random as RD
import numpy as np
import pickle


RD.seed()
np.random.seed()

do_computations = False
save_computations = False
load_computations = True
assert (not save_computations) or do_computations, "do_computations should be true to save_computations"

do_plots = True

save_plots = True

show_plots = False

assert show_plots is not save_plots, "you can either save plots or show plots not both"

simulator_mode = False

layout = 'spring'  # layout could be circular, spring

if simulator_mode:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    import pickle

if do_plots:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    from decimal import Decimal
    FOURPLACES = Decimal(10) ** -4
    TWOPLACES = Decimal(10) ** -2
    ERROR_BAR_TYPE = 'std'

def combine(list_of_names,output_name):
    '''Useful for combining outputs from cluster computations.'''
    original = pickle.load(open('./data/' + list_of_names[0] + '.pkl', 'rb'))
    for ii in range(1, len(list_of_names)):
        additional = pickle.load(open('./data/' + list_of_names[ii] + '.pkl', 'rb'))
        print(original)
        print(additional)
        original = np.concatenate((original,additional),axis=1)
        print(original)
    pickle.dump(original, open('./data/'+output_name+'.pkl', 'wb'))
