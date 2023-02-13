# goes over pickled spreading times and puts them into the CSV file: network_group + 'properties_data_dump.csv'

from settings import *

from models import measure_property

import pandas as pd

intervention_size_list = [10] #[5, 10, 15, 20, 25]

old_properties = ['avg_clustering', 'average_shortest_path_length', 'diameter', 'size_2_core']
# if for dumping the fb100 data, it should be:
# old_properties = ['avg_clustering']

new_properties = ['avg_degree', 'diam_2_core', 'max_degree', 'min_degree',
                  'max_degree_2_core', 'min_degree_2_core',
                  'avg_degree_2_core', 'number_edges','number_edges_2_core',
                  'avg_clustering_2_core', 'transitivity', 'transitivity_2_core',
                  'num_leaves']

all_properties = old_properties + new_properties
# if for dumping the fb100 data, it should be:
# all_properties = old_properties

included_properties = all_properties

include_original_networks = True
include_rewiring_networks = True
include_addition_networks = True

if __name__ == '__main__':

    assert data_dump and load_computations, "we should be in data_dump and load_computations mode!"

    generating_new_dump = False

    try:
        df = pd.read_csv(output_directory_address + network_group + 'properties_data_dump.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                   'intervention_type',
                                   'intervention_size']+included_properties, dtype='float')
        print('New ' + network_group + 'clustering_data_dump file will be generated.')
        generating_new_dump = True

    assert generating_new_dump, "Need a new data dump!"

    if include_original_networks:

        #  dump the properties of the original networks (no interventions):

        original_properties = [[] for ii in range(len(included_properties))]

        for network_id in network_id_list:

            print(network_id)

            #  load in the network and extract preliminary data

            fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

            G = NX.read_edgelist(fh, delimiter=DELIMITER)

            print('original size ', len(G.nodes()))

            #  get the largest connected component:
            if not NX.is_connected(G):
                G = max(NX.connected_component_subgraphs(G), key=len)
                print('largest connected component extracted with size ', len(G.nodes()))

            #  remove self loops:
            if len(list(NX.selfloop_edges(G))) > 0:
                print('warning the graph has ' + str(len(list(NX.selfloop_edges(G)))) + ' self-loops that will be removed')
                print('number of edges before self loop removal: ', G.size())
                G.remove_edges_from(NX.selfloop_edges(G))
                print('number of edges before self loop removal: ', G.size())

            network_size = NX.number_of_nodes(G)

            G_list = [G]
            for included_property in included_properties:
                original_properties[included_properties.index(included_property)] += \
                    measure_property(G_list, included_property)
                print(included_property, original_properties[included_properties.index(included_property)][-1])
            print(network_id_list.index(network_id))
            print(original_properties)
            print([this_property[network_id_list.index(network_id)] for this_property in original_properties])

            print('we are in data_dump mode')

            dataset = [[network_group, network_id, network_size,
                        'none', 0.0]
                       + [this_property[network_id_list.index(network_id)] for this_property in original_properties]]

            new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
                                                         'intervention_type',
                                                         'intervention_size'] + included_properties)

            print(new_df)

            extended_frame = [df, new_df]  # , df_add_triad, df_rewired, df_original]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
            drop=True)

    # interventions

    for intervention_size in intervention_size_list:

        print('intervention size:', intervention_size)

        mean_properties_add_random = [[] for i in range(len(included_properties))]
        mean_properties_add_triad = [[] for i in range(len(included_properties))]
        mean_properties_rewired = [[] for i in range(len(included_properties))]

        std_properties_add_random = [[] for i in range(len(included_properties))]
        std_properties_add_triad = [[] for i in range(len(included_properties))]
        std_properties_rewired = [[] for i in range(len(included_properties))]

        for network_id in network_id_list:

            print('network id:', network_id)

            #  load in the network and extract preliminary data

            fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

            G = NX.read_edgelist(fh, delimiter=DELIMITER)

            print('original size ', len(G.nodes()))

            if not NX.is_connected(G):

                G = max(NX.connected_component_subgraphs(G), key=len)
                print('largest connected component extracted with size ', len(G.nodes()))

            if len(list(NX.selfloop_edges(G))) > 0:
                print(
                    'warning the graph has ' + str(len(list(NX.selfloop_edges(G)))) + ' self-loops that will be removed.')
                print('number of edges before self loop removal: ', G.size())
                G.remove_edges_from(NX.selfloop_edges(G))
                print('number of edges before self loop removal: ', G.size())

            network_size = NX.number_of_nodes(G)

            # load properties:
            properties_sample_add_random = []
            properties_sample_add_triad = []
            properties_sample_rewired = []

            for included_property in included_properties:

                if include_addition_networks:

                    properties_sample_add_random += \
                        [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'add_random_'
                                          + network_group + network_id + '.pkl', 'rb'))]

                    properties_sample_add_triad += \
                        [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'add_triad_'
                                          + network_group + network_id + '.pkl', 'rb'))]

                    mean_properties_add_random[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_add_random[-1])]

                    mean_properties_add_triad[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_add_triad[-1])]

                    std_properties_add_random[included_properties.index(included_property)] \
                        += [np.std(properties_sample_add_random[-1])]

                    std_properties_add_triad[included_properties.index(included_property)] \
                        += [np.std(properties_sample_add_triad[-1])]

                if include_rewiring_networks:

                    properties_sample_rewired += \
                        [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'rewiring_'
                                          + network_group + network_id + '.pkl', 'rb'))]



                    mean_properties_rewired[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_rewired[-1])]



                    std_properties_rewired[included_properties.index(included_property)] \
                        += [np.std(properties_sample_rewired[-1])]

                print('loaded ' + included_property + '_'
                      + str(intervention_size) + '_percent_' + 'interventions_'
                      + network_group + network_id)

                print('loaded all properties.')

            # dumping the loaded data:

            dataset = []

            if include_addition_networks:
                dataset += [[network_group, network_id, network_size,
                            'random_addition', intervention_size]
                            + [this_property[network_id_list.index(network_id)]
                               for this_property in mean_properties_add_random],
                            [network_group, network_id, network_size,
                            'triad_addition', intervention_size]
                            + [this_property[network_id_list.index(network_id)]
                               for this_property in mean_properties_add_triad]]

            if include_rewiring_networks:
                dataset += [[network_group, network_id, network_size,
                            'rewiring', intervention_size]
                            + [this_property[network_id_list.index(network_id)]
                               for this_property in mean_properties_rewired]]


            # dataset = [[network_group, network_id, network_size,
            #             'random_addition', intervention_size]
            #            + [this_property[network_id_list.index(network_id)]
            #               for this_property in mean_properties_add_random],
            #            [network_group, network_id, network_size,
            #             'triad_addition', intervention_size]
            #            + [this_property[network_id_list.index(network_id)]
            #               for this_property in mean_properties_add_triad],
            #            [network_group, network_id, network_size,
            #             'rewiring', intervention_size]
            #            + [this_property[network_id_list.index(network_id)]
            #               for this_property in mean_properties_rewired]]

            new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
                                                         'intervention_type',
                                                         'intervention_size'] + included_properties)

            print(new_df)

            extended_frame = [df, new_df] #, df_add_triad, df_rewired, df_original]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
                drop=True)

    df.to_csv(output_directory_address + network_group + 'properties_data_dump.csv', index=False)#  , index=False