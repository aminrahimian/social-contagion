# Comparing the rate of contagion in the original and rewired network.
# uses maslov_sneppen_rewiring(num_steps = np.floor(0.1 * self.params['network'].number_of_edges()))
# to rewire the network
# uses avg_speed_of_spread(dataset_size=1000,cap=0.9, mode='max') to measure the rate of spread

from settings import *
import settings


# assert settings.data_dump, "we should be in data_dump mode!"

from models import *

size_of_dataset = 200

intervention_size_list = [5, 10, 15, 20, 25]

included_properties = ['avg_clustering', 'average_shortest_path_length', 'diameter', 'size_2_core']
    # ['size_2_core']#['average_shortest_path_length','diameter']#['avg_clustering', 'average_shortest_path_length', 'diameter', 'size_2_core']

generate_network_intervention_dataset = False

use_separate_address_for_pickled_networks = True  # pickled_networks take a lot of space.
# Some may need to put them else where away from other pickled samples.

separate_address_for_pickled_networks = '/home/amin/Desktop/pickled_networks/'

assert (generate_network_intervention_dataset is None) or settings.do_computations, \
    'generate_network_intervention_dataset cannot be set (True or False)  when not do_computations'


assert (not (generate_network_intervention_dataset is None)) or (not settings.do_computations), \
    'generate_network_intervention_dataset should not be None when do_computations'

if __name__ == '__main__':

    if data_dump:
        try:
            df = pd.read_csv(output_directory_address + network_group + 'properties_data_dump.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                       'number_edges', 'intervention_type',
                                       'intervention_size']+included_properties, dtype='float')
            print('New ' + network_group + 'clustering_data_dump file will be generated.')

    # no interventions the original network:

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
        if len(list(G.selfloop_edges())) > 0:
            print('warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
            print('number of edges before self loop removal: ', G.size())
            G.remove_edges_from(G.selfloop_edges())
            print('number of edges before self loop removal: ', G.size())

        network_size = NX.number_of_nodes(G)

        number_edges = G.number_of_edges()

        G_list = [G]
        for included_property in included_properties:
            original_properties[included_properties.index(included_property)] += \
                measure_property(G_list, included_property)
            print(included_property, original_properties[included_properties.index(included_property)][-1])
        print(network_id_list.index(network_id))
        print(original_properties)
        print([this_property[network_id_list.index(network_id)] for this_property in original_properties])

        if settings.data_dump:
            print('we are in data_dump mode')

            dataset = [[network_group, network_id, network_size,
                        number_edges, 'none', 0.0]
                       + [this_property[network_id_list.index(network_id)] for this_property in original_properties]]

            new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
                                                         'number_edges', 'intervention_type',
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

            if len(list(G.selfloop_edges())) > 0:
                print(
                    'warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed.')
                print('number of edges before self loop removal: ', G.size())
                G.remove_edges_from(G.selfloop_edges())
                print('number of edges before self loop removal: ', G.size())

            network_size = NX.number_of_nodes(G)

            number_edges = G.number_of_edges()

            if settings.do_computations:

                number_initial_seeds = 2

                params_add_random = {
                    'network': G,
                    'size': network_size,
                    'add_edges': True,
                    'edge_addition_mode': 'random',
                    'number_of_edges_to_be_added': int(np.floor(0.01*intervention_size * G.number_of_edges())),
                    # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                    'initialization_mode': 'fixed_number_initial_infection',
                    'initial_infection_number': number_initial_seeds,
                    'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                    'fixed_prob_high': 1.0,
                    'fixed_prob': 0.05,
                    'theta': 2,
                    'rewire': False,
                }

                dynamics_add_random = DeterministicLinear(params_add_random)

                params_add_triad = {
                    'network': G,
                    'size': network_size,
                    'add_edges': True,
                    'edge_addition_mode': 'triadic_closures',
                    'number_of_edges_to_be_added': int(np.floor(0.01 * intervention_size * G.number_of_edges())),
                    # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                    'initialization_mode': 'fixed_number_initial_infection',
                    'initial_infection_number': number_initial_seeds,
                    'delta': 0.0000000000000001,
                    'fixed_prob_high': 1.0,
                    'fixed_prob': 0.05,
                    'theta': 2,
                    'rewire': False,
                    # rewire 10% of edges
                }

                dynamics_add_triad = DeterministicLinear(params_add_triad)

                params_rewired = {
                    'network': G,
                    'size': network_size,
                    'add_edges': False,
                    # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                    'initialization_mode': 'fixed_number_initial_infection',
                    'initial_infection_number': number_initial_seeds,
                    'delta': 0.0000000000000001,
                    'fixed_prob_high': 1.0,
                    'fixed_prob': 0.05,
                    'theta': 2,
                    'rewire': True,
                    'rewiring_mode': 'random_random',
                    'num_edges_for_random_random_rewiring': 0.01 * intervention_size * G.number_of_edges(),
                    # rewire 15% of edges
                }

                dynamics_rewired = DeterministicLinear(params_rewired)

                if generate_network_intervention_dataset is True:

                    add_random_networks = \
                        dynamics_add_random.generate_network_intervention_dataset(dataset_size=size_of_dataset)

                    print('generated add_random_networks', add_random_networks)

                    add_triad_networks = \
                        dynamics_add_random.generate_network_intervention_dataset(dataset_size=size_of_dataset)

                    print('generated add_traid_networks', add_triad_networks)

                    rewired_networks = \
                        dynamics_rewired.generate_network_intervention_dataset(dataset_size=size_of_dataset)

                    print('generated rewired_networks', rewired_networks)

                elif generate_network_intervention_dataset is False:

                    if use_separate_address_for_pickled_networks:
                        add_random_networks = pickle.load(open(separate_address_for_pickled_networks + 'networks_'
                                                               + str(intervention_size) + '_percent_' + 'add_random_'
                                                               + network_group + network_id + '.pkl', 'rb'))
                        print('loaded add_random_networks from separate address', add_random_networks)

                        add_triad_networks = pickle.load(open(separate_address_for_pickled_networks + 'networks_'
                                                              + str(intervention_size) + '_percent_' + 'add_triad_'
                                                              + network_group + network_id + '.pkl', 'rb'))

                        print('loaded add_triad_networks from separate address', add_triad_networks)

                        rewired_networks = pickle.load(open(separate_address_for_pickled_networks + 'networks_'
                                                            + str(intervention_size) + '_percent_' + 'rewiring_'
                                                            + network_group + network_id + '.pkl', 'rb'))

                        print('loaded rewired_networks from separate address', rewired_networks)

                    else:
                        add_random_networks = pickle.load(open(pickled_samples_directory_address + 'networks_'
                                                               + str(intervention_size) + '_percent_' + 'add_ran_'
                                                               + network_group + network_id + '.pkl', 'rb'))
                        print('loaded add_random_networks from common pickle address', add_random_networks)

                        add_triad_networks = pickle.load(open(pickled_samples_directory_address + 'networks_'
                                                              + str(intervention_size) + '_percent_' + 'add_triad_'
                                                              + network_group + network_id + '.pkl', 'rb'))

                        print('loaded add_triad_networks from common pickle address', add_triad_networks)

                        rewired_networks = pickle.load(open(pickled_samples_directory_address + 'networks_'
                                                            + str(intervention_size) + '_percent_' + 'rewiring_'
                                                            + network_group + network_id + '.pkl', 'rb'))

                        print('loaded rewired_networks from common pickle address', rewired_networks)

                    properties_sample_add_random = []
                    properties_sample_add_triad = []
                    properties_sample_rewired = []

                    for included_property in included_properties:

                        properties_sample_add_random += [measure_property(add_random_networks, included_property)]
                        properties_sample_add_triad += [measure_property(add_triad_networks, included_property)]
                        properties_sample_rewired += [measure_property(rewired_networks, included_property)]

                        # print(properties_sample_add_random)
                        #
                        # print(properties_sample_add_random[-1])

                        mean_properties_add_random[included_properties.index(included_property)] \
                            += [np.mean(properties_sample_add_random[-1])]

                        mean_properties_add_triad[included_properties.index(included_property)] \
                            += [np.mean(properties_sample_add_triad[-1])]

                        mean_properties_rewired[included_properties.index(included_property)] \
                            += [np.mean(properties_sample_rewired[-1])]

                        std_properties_add_random[included_properties.index(included_property)] \
                            += [np.std(properties_sample_add_random[-1])]

                        std_properties_add_triad[included_properties.index(included_property)] \
                            += [np.std(properties_sample_add_triad[-1])]

                        std_properties_rewired[included_properties.index(included_property)] \
                            += [np.std(properties_sample_rewired[-1])]

                        # print(included_property,
                        #       mean_properties_add_random[included_properties.index(included_property)],
                        #       std_properties_add_random[included_properties.index(included_property)])
                        #
                        # print(properties_sample_add_random[-1])
                        # print(mean_properties_add_random)
                        # print(std_properties_add_random)

                else:
                    assert False, "generate_network_intervention_dataset is not set properly for do_computations."

            if settings.save_computations:

                if generate_network_intervention_dataset is True:

                    if use_separate_address_for_pickled_networks:

                        pickle.dump(add_random_networks,
                                    open(separate_address_for_pickled_networks+ 'networks_'
                                         + str(intervention_size) + '_percent_' + 'add_random_'
                                         + network_group + network_id + '.pkl', 'wb'))
                        pickle.dump(add_triad_networks,
                                    open(separate_address_for_pickled_networks + 'networks_'
                                         + str(intervention_size) + '_percent_' + 'add_triad_'
                                         + network_group + network_id + '.pkl', 'wb'))

                        pickle.dump(rewired_networks,
                                    open(separate_address_for_pickled_networks + 'networks_'
                                         + str(intervention_size) + '_percent_' + 'rewiring_'
                                         + network_group + network_id + '.pkl', 'wb'))

                        print('dumped networks in a separate address')

                    else:

                        pickle.dump(add_random_networks,
                                    open(pickled_samples_directory_address + 'networks_'
                                         + str(intervention_size) + '_percent_' + 'add_random_'
                                         + network_group + network_id + '.pkl', 'wb'))

                        pickle.dump(add_triad_networks,
                                    open(pickled_samples_directory_address + 'networks_'
                                         + str(intervention_size) + '_percent_' + 'add_triad_'
                                         + network_group + network_id + '.pkl', 'wb'))

                        pickle.dump(rewired_networks,
                                    open(pickled_samples_directory_address + 'networks_'
                                         + str(intervention_size) + '_percent_' + 'rewiring_'
                                         + network_group + network_id + '.pkl', 'wb'))

                        print('dumped networks in the common pickle address')

                elif generate_network_intervention_dataset is False:

                    for included_property in included_properties:
                        print(included_property, properties_sample_add_random[included_properties.index(included_property)])
                        pickle.dump(properties_sample_add_random[included_properties.index(included_property)],
                                    open(pickled_samples_directory_address + included_property + '_'
                                         + str(intervention_size) + '_percent_' + 'add_random_'
                                         + network_group + network_id + '.pkl', 'wb'))
                        print(included_property,
                              properties_sample_add_triad[included_properties.index(included_property)])
                        pickle.dump(properties_sample_add_triad[included_properties.index(included_property)],
                                    open(pickled_samples_directory_address + included_property + '_'
                                         + str(intervention_size) + '_percent_' + 'add_triad_'
                                         + network_group + network_id + '.pkl', 'wb'))
                        print(included_property,
                              properties_sample_rewired[included_properties.index(included_property)])
                        pickle.dump(properties_sample_rewired[included_properties.index(included_property)],
                                    open(pickled_samples_directory_address + included_property + '_'
                                         + str(intervention_size) + '_percent_' + 'rewiring_'
                                         + network_group + network_id + '.pkl', 'wb'))
                        print('dumped ' + included_property + '_'
                              + str(intervention_size) + '_percent_' + 'interventions_'
                              + network_group + network_id)

                    print('dumped all properties.')
                else:
                    assert False, "generate_network_intervention_dataset is not set properly for do_computations."

            if settings.load_computations:

                properties_sample_add_random = []
                properties_sample_add_triad = []
                properties_sample_rewired = []

                for included_property in included_properties:

                    properties_sample_add_random += \
                        [pickle.load(open(pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'add_random_'
                                          + network_group + network_id + '.pkl', 'rb'))]

                    properties_sample_add_triad += \
                        [pickle.load(open(pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'add_triad_'
                                          + network_group + network_id + '.pkl', 'rb'))]

                    properties_sample_rewired += \
                        [pickle.load(open(pickled_samples_directory_address + included_property + '_'
                                          + str(intervention_size) + '_percent_' + 'rewiring_'
                                          + network_group + network_id + '.pkl', 'rb'))]

                    mean_properties_add_random[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_add_random[-1])]

                    mean_properties_add_triad[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_add_triad[-1])]

                    mean_properties_rewired[included_properties.index(included_property)] \
                        += [np.mean(properties_sample_rewired[-1])]

                    std_properties_add_random[included_properties.index(included_property)] \
                        += [np.std(properties_sample_add_random[-1])]

                    std_properties_add_triad[included_properties.index(included_property)] \
                        += [np.std(properties_sample_add_triad[-1])]

                    std_properties_rewired[included_properties.index(included_property)] \
                        += [np.std(properties_sample_rewired[-1])]


                    print('loaded ' + included_property + '_'
                          + str(intervention_size) + '_percent_' + 'interventions_'
                          + network_group + network_id)

                # pickle.dump(clustering_samples_add_random, open(pickled_samples_directory_address + 'clustering_samples_'
                #                                                 + str(intervention_size) + '_percent_' + 'add_random_'
                #                                                 + network_group + network_id + '.pkl', 'wb'))
                # pickle.dump(clustering_samples_add_triad, open(pickled_samples_directory_address + 'clustering_samples_'
                #                                                + str(intervention_size) + '_percent_' + 'add_triad_'
                #                                                + network_group + network_id + '.pkl', 'wb'))
                # pickle.dump(clustering_samples_rewired,
                #             open(pickled_samples_directory_address + 'clustering_samples_'
                #                  + str(intervention_size) + '_percent_' + 'rewiring_'
                #                  + network_group + network_id + '.pkl', 'wb'))
                print('loaded all properties.')

                # clustering_samples_add_random = pickle.load(open(pickled_samples_directory_address + 'clustering_samples_'
                #                                                  + str(intervention_size) + '_percent_' + 'add_random_'
                #                                                  + network_group + network_id + '.pkl', 'rb'))
                #
                # clustering_samples_add_triad = pickle.load(open(pickled_samples_directory_address + 'clustering_samples_'
                #                                                 + str(intervention_size) + '_percent_' + 'add_triad_'
                #                                                 + network_group + network_id + '.pkl', 'rb'))
                #
                # clustering_samples_rewired = pickle.load(open(pickled_samples_directory_address + 'clustering_samples_'
                #                                               + str(intervention_size) + '_percent_' + 'rewiring_'
                #                                               + network_group + network_id + '.pkl', 'rb'))
                #
                # clustering_add_triad = np.mean(clustering_samples_add_triad)
                #
                # clustering_add_random = np.mean(clustering_samples_add_random)
                #
                # clustering_rewired = np.mean(clustering_samples_rewired)
                #
                # clustering_std_add_triad = np.std(clustering_samples_add_triad)
                #
                # clustering_std_add_random = np.std(clustering_samples_add_random)
                #
                # clustering_std_rewired = np.std(clustering_samples_rewired)


            # if settings.do_plots:
            #
            #     avg_clustering_add_random += [clustering_add_random]
            #     avg_clustering_add_triad += [clustering_add_triad]
            #     avg_clustering_rewired += [clustering_rewired]
            #
            #     std_clustering_add_random += [clustering_std_add_random]
            #     std_clustering_add_triad += [clustering_std_add_triad]
            #     std_clustering_rewired += [clustering_std_rewired]

            if settings.data_dump:

                print('we are in data_dump mode')

                dataset = [[network_group, network_id, network_size,
                            number_edges, 'random_addition', intervention_size]
                           + [this_property[network_id_list.index(network_id)]
                              for this_property in mean_properties_add_random],
                           [network_group, network_id, network_size,
                            number_edges, 'triad_addition', intervention_size]
                           + [this_property[network_id_list.index(network_id)]
                              for this_property in mean_properties_add_triad],
                           [network_group, network_id, network_size,
                            number_edges, 'rewiring', intervention_size]
                           + [this_property[network_id_list.index(network_id)]
                              for this_property in mean_properties_rewired]]

                new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
                                                             'number_edges', 'intervention_type',
                                                             'intervention_size'] + included_properties)

                print(new_df)

                extended_frame = [df, new_df] #, df_add_triad, df_rewired, df_original]

                df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
                    drop=True)

        if settings.do_plots:

            for included_property in included_properties:

                plt.figure()

                iii = included_properties.index(included_property)

                plt.hist([original_properties[iii],
                          mean_properties_add_random[iii],
                          mean_properties_add_triad[iii],
                          mean_properties_rewired[iii]],
                         label=['original',
                                str(intervention_size) + ' \% random edge addition',
                                str(intervention_size) + ' \% triad edge addition',
                                str(intervention_size) + ' \% rewiring'])

                plt.ylabel('Frequency')
                plt.xlabel(included_property)
                plt.title(included_property + ' under Various Interventions, Intervention Size: '
                          + str(intervention_size) + '%')
                plt.legend()
                if settings.show_plots:
                    plt.show()
                if settings.save_plots:
                    plt.savefig(output_directory_address + included_property + '_'
                                + 'intervention_size_' + str(intervention_size)+'.png')

    if settings.data_dump:
        df.to_csv(output_directory_address + network_group + 'properties_data_dump.csv', index=False)#  , index=False