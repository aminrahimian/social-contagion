# Params and other settings are set here
# Settings are for the generative model as well as the inference engine

# The generative model settings

import random as RD
import numpy as np
import pickle
import os
import errno
import networkx as NX
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text) ]

network_group = 'banerjee_combined_edgelist_'
#'cai_edgelist_' #'fb100_edgelist_'
# 'banerjee_combined_edgelist_'
#'chami_friendship_edgelist_'
#'chami_advice_edgelist_'
#'cai_edgelist_'

if network_group == 'cai_edgelist_':

    root_data_address = './data/cai-data/'

    DELIMITER = ' '


elif network_group == 'chami_friendship_edgelist_':

    root_data_address = './data/chami-friendship-data/'

    DELIMITER = ','


elif network_group == 'chami_advice_edgelist_':

    root_data_address = './data/chami-advice-data/'

    DELIMITER = ','

elif network_group == 'banerjee_combined_edgelist_':

    root_data_address = './data/banerjee-combined-data/'

    DELIMITER = ' '

elif network_group == 'fb100_edgelist_':

    root_data_address = './data/fb100-data/'

    DELIMITER = ' '

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

pickled_samples_directory_address = root_data_address + 'pickled_samples/'

properties_pickled_samples_directory_address = pickled_samples_directory_address + 'properties_pickled_samples/'

spreading_pickled_samples_directory_address = pickled_samples_directory_address + 'spreading_pickled_samples/'

networks_pickled_samples_directory_address = pickled_samples_directory_address + 'networks_pickled_samples/'
use_separate_address_for_pickled_networks = False  # pickled_networks take a lot of space.
# Some may need to put them else where with a lot of space away from other pickled samples.
separate_address_for_pickled_networks = '/home/rahimian/contagion/data/pickled_networks/'
# '/home/rahimian/contagion/data/pickled_networks/'
#  '/home/amin/Desktop/pickled_networks/'
if use_separate_address_for_pickled_networks:
    networks_pickled_samples_directory_address = \
        separate_address_for_pickled_networks

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

try:
    os.makedirs(properties_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(spreading_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(networks_pickled_samples_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

network_id_list = []
for file in os.listdir(edgelist_directory_address):
    filename = os.path.splitext(file)[0]
    net_id = filename.replace(network_group,'')
    print(net_id)
    network_id_list += [net_id]
network_id_list.sort(key=natural_keys)
print(network_id_list)


#  different models:
model_id = '_model_1'

if model_id == '_model_1':
    MODEL = '(0.05,1)'
    fixed_prob_high = 1.0
    fixed_prob_low = 0.05
    alpha = 1.0
    gamma = 0.0
elif model_id == '_model_2':
    MODEL = '(0.05,0.5)'
    fixed_prob_high = 0.5
    fixed_prob_low = 0.05
    alpha = 1.0
    gamma = 0.0
elif model_id == '_model_3':
    MODEL = '(0.05,1(0.05,0.5))'
    fixed_prob_high = 1.0
    fixed_prob_low = 0.05
    alpha = 0.05
    gamma = 0.5
else:
    print('model_id is not valid')
    exit()



#  different modes of operation:
#
# do_computations = False
# save_computations = False
# load_computations = True
# do_plots = False
# save_plots = False
# show_plots = False
# data_dump = True
# simulator_mode = False

do_computations = True
do_multiprocessing = True
save_computations = True
load_computations = False
do_plots = False
save_plots = False
show_plots = False
data_dump = False
simulator_mode = False


#  check that different modes are set consistently

assert (not save_computations) or do_computations, "do_computations should be true to save_computations"

assert (not do_multiprocessing) or do_computations, "do_computations should be true to do_multiprocessing"

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

    simulator_ID = 'cai_edgelist_1'

    root_data_address = './data/cai-data/'

    edgelist_directory_address = root_data_address + 'edgelists/'

    pickled_samples_directory_address = root_data_address + 'pickled_samples/'

    output_directory_address = root_data_address + 'output/'

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

    fh = open(edgelist_directory_address + simulator_ID + '.txt', 'rb')

    G = NX.read_edgelist(fh)

    print(NX.is_connected(G))

    network_size = NX.number_of_nodes(G)

    initial_seeds = 2

    simulator_params = {
        'network': G,
        # 'size': network_size,  # populationSize,
        'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
        # two initial seeds, next to each other
        'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
        # 'nearest_neighbors': 4,
        # 'fixed_number_edges_added': 2,
        'fixed_prob_high': 1.0,
        'fixed_prob': 0.05,
        'theta': 2,
        'alpha': 0.05,
        'gamma': 0.5,
    }

if do_plots:
    import matplotlib.pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    from decimal import Decimal
    FOURPLACES = Decimal(10) ** -4
    TWOPLACES = Decimal(10) ** -2
    ERROR_BAR_TYPE = 'std'

if do_multiprocessing:
    import multiprocessing
    from itertools import product
    number_CPU = 13


def combine(list_of_names,output_name):
    '''Useful for combining pickle files from cluster computations.'''
    original = pickle.load(open('./data/' + list_of_names[0] + '.pkl', 'rb'))
    for ii in range(1, len(list_of_names)):
        additional = pickle.load(open('./data/' + list_of_names[ii] + '.pkl', 'rb'))
        print(original)
        print(additional)
        original = np.concatenate((original,additional),axis=1)
        print(original)
    pickle.dump(original, open('./data/'+output_name+'.pkl', 'wb'))
