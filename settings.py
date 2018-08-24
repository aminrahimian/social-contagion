# Params and other settings are set here
# Settings are for the generative model as well as the inference engine

# The generative model settings

import random as RD
import numpy as np
import pickle
import os
import errno

RD.seed()
np.random.seed()

network_group = 'banerjee_combined_edgelist_' #'banerjee_combined_edgelist_'

if network_group == 'cai_edgelist_':

    root_data_address = './data/cai-data/'

    DELIMITER = ' '

    TOP_ID = 175

elif network_group == 'chami_friendship_edgelist_':

    root_data_address = './data/chami-friendship-data/'

    DELIMITER = ','

    TOP_ID = 17

elif network_group == 'chami_advice_edgelist_':

    root_data_address = './data/chami-advice-data/'

    DELIMITER = ','

    TOP_ID = 17

elif network_group == 'banerjee_combined_edgelist_':

    root_data_address = './data/banerjee-combined-data/'

    DELIMITER = ' '

    TOP_ID = 77

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

pickled_samples_directory_address = root_data_address + 'pickled_samples/'

try:
    os.makedirs(output_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


network_id_list = list(np.linspace(1,TOP_ID,TOP_ID))

if network_group == 'banerjee_combined_edgelist_':
    del network_id_list[12]
    del network_id_list[21]

network_id_list = [str(int(id)) for id in network_id_list]


#  different models:
model_id = '_model_1'

if model_id == '_model_1':
    MODEL = '(0.05,1)'
    fixed_prob_high = 1.0
    fixed_prob_low = 0.05
elif model_id == '_model_2':
    MODEL = '(0.05,0.5)'
    fixed_prob_high = 0.5
    fixed_prob_low = 0.05
elif model_id == '_model_3':
    MODEL = '(SIRI(0.5,0.05),1)'
    fixed_prob_high = 1.0
else:
    print('model_id is not valid')
    exit()

number_initial_seeds = 2

#  different modes of operation:

do_computations = True
save_computations = True
load_computations = False
do_plots = False
save_plots = False
show_plots = False
data_dump = False
simulator_mode = False

#  check that different modes are set consistently

assert (not save_computations) or do_computations, "do_computations should be true to save_computations"

assert (not (save_plots or show_plots)) or do_plots, "do_plots should be true to save_plots or show_plots"

assert (not do_plots) or save_plots or show_plots, "when do_plots should either save_plots or show_plots"

assert not (load_computations and save_computations), "cannot both save_computations and load_computations"

assert not (load_computations and do_computations), "cannot both do_computations and load_computations"

assert not (show_plots and save_plots), "you can either save plots or show plots not both"

assert not (data_dump and (save_plots or show_plots)), "you cannot do any plots in data_dump mode"

assert not (data_dump and (save_computations or do_computations)), "you cannot do any computations in data_dump mode"

assert (not data_dump) or load_computations, "load_computations should be true to data_dump"

assert not (simulator_mode and do_plots), "simulator_mode and do_plots use different " \
                                          "(conflicting) matplotlib settings, and " \
                                          "cannot be both true"

layout = 'spring'  # layout could be circular, spring, this the graph visualization layout

# import the required packages for different modes:

if data_dump:
    import csv
    import pandas as pd

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
