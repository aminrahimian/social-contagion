# Takes Chami friendship and advice network and generates their union

import networkx as NX

import os

import errno

friendship_network_group = 'chami_friendship_edgelist_'

advice_network_group = 'chami_advice_edgelist_'

union_network_group = 'chami_union_edgelist_'

DELIMITER = ','

chami_friendship_root_data_address = './data/chami-friendship-data/'

chami_friendship_DELIMITER = ','

chami_advice_root_data_address = './data/chami-advice-data/'

chami_advice_DELIMITER = ','

chami_friendship_edgelist_directory_address = chami_friendship_root_data_address + 'edgelists/'

chami_advice_edgelist_directory_address = chami_advice_root_data_address + 'edgelists/'

chami_union_root_data_address = './data/chami-union-data/'

chami_union_DELIMITER = ','

chami_union_edgelist_directory_address = chami_union_root_data_address + 'edgelists/'

try:
    os.makedirs(chami_union_root_data_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

try:
    os.makedirs(chami_union_edgelist_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

network_id_list = []

friendship_filenames = []

advice_filenames = []

for file in os.listdir(chami_friendship_edgelist_directory_address):
    friendship_filenames += [os.path.splitext(file)[0]]

for file in os.listdir(chami_advice_edgelist_directory_address):
    advice_filenames += [os.path.splitext(file)[0]]

assert len(friendship_filenames) == len(advice_filenames), \
    "numbers of friendship and advice networks to be combined do not match."

for id in range(len(friendship_filenames)):

    net_id = friendship_filenames[id].replace(friendship_network_group, '')

    print(net_id)

    # loading friendship network

    fh_friendship = open(chami_friendship_edgelist_directory_address
                         + friendship_network_group
                         + net_id + '.txt', 'rb')

    G_friendship = NX.read_edgelist(fh_friendship, delimiter=chami_friendship_DELIMITER)

    # loading advice network

    fh_advice = open(chami_advice_edgelist_directory_address
                     + advice_network_group
                     + net_id + '.txt', 'rb')

    G_advice = NX.read_edgelist(fh_friendship, delimiter=chami_advice_DELIMITER)

    # compose the union network

    G_union = NX.compose(G_friendship, G_advice)

    # write the composed network

    NX.write_edgelist(G_union,
                      chami_union_edgelist_directory_address
                      + union_network_group
                      + net_id + '.txt',
                      delimiter=chami_union_DELIMITER, data=False)