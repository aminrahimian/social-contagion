# Comparing the rate of contagion in the original and rewired network.
# uses maslov_sneppen_rewiring(num_steps = np.floor(0.1 * self.params['network'].number_of_edges()))
# to rewire the network
# uses avg_speed_of_spread(dataset_size=1000,cap=0.9, mode='max') to measure the rate of spread

from settings import *
import settings
import os
import errno

# assert settings.data_dump, "we should be in data_dump mode!"

from models import *

size_of_dataset = 200

network_group = 'cai_edgelist_'

root_data_address = './data/cai-data/'

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

try:
    os.makedirs(output_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


network_id_list = list(np.linspace(1,175,175))

network_id_list = [str(int(id)) for id in network_id_list]

percent_more_edges = 15

rewiring_percentage = 5

if __name__ == '__main__':

    if data_dump:
        try:
            df = pd.read_csv(output_directory_address + network_group + 'clustering_data_dump.csv')
        except FileNotFoundError:
            df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                                         'number_edges', 'intervention_type',
                                                         'intervention_size', 'avg_clustering'],dtype='float')
            print('New ' + network_group + 'clustering_data_dump file will be generated.')

    avg_clustering_add_random = []
    avg_clustering_add_triad = []
    avg_clustering_rewired = []

    std_clustering_add_random = []
    std_clustering_add_triad = []
    std_clustering_rewired = []

    clustering_original = []

    for network_id in network_id_list:

        print(network_id)

        #  load in the network and extract preliminary data

        fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

        G = NX.read_edgelist(fh)

        print(NX.is_connected(G))

        network_size = NX.number_of_nodes(G)

        number_edges = G.number_of_edges()

        original_average_clustering = NX.average_clustering(G)

        if settings.do_computations:

            number_initial_seeds = 2

            params_add_random = {
                'network': G,
                'size': network_size,
                'add_edges': True,
                'edge_addition_mode': 'random',
                'number_of_edges_to_be_added': int(np.floor(0.01*percent_more_edges * G.number_of_edges())),
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

            clustering_samples_add_random = \
                dynamics_add_random.measure_avg_clustering(dataset_size=size_of_dataset)

            clustering_add_random = np.mean(clustering_samples_add_random)

            clustering_std_add_random = np.std(clustering_samples_add_random)

            print(clustering_add_random, clustering_std_add_random)

            print(clustering_samples_add_random)

            params_add_triad = {
                'network': G,
                'size': network_size,
                'add_edges': True,
                'edge_addition_mode': 'triadic_closures',
                'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
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

            dynamics_add_triad = DeterministicLinear(params_add_triad)

            clustering_samples_add_triad = \
                dynamics_add_triad.measure_avg_clustering(dataset_size=size_of_dataset)

            clustering_add_triad = np.mean(clustering_samples_add_triad)

            clustering_std_add_triad = np.std(clustering_samples_add_triad)

            print(clustering_add_triad, clustering_std_add_triad)

            print(clustering_samples_add_triad)

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
                'num_edges_for_random_random_rewiring': 0.01 * rewiring_percentage * G.number_of_edges(),
                # rewire 15% of edges
            }

            dynamics_rewired = DeterministicLinear(params_rewired)

            clustering_samples_rewired = \
                dynamics_rewired.measure_avg_clustering(dataset_size=size_of_dataset)

            clustering_rewired = np.mean(clustering_samples_rewired)

            clustering_std_rewired = np.std(clustering_samples_rewired)

            print(clustering_rewired, clustering_std_rewired)

            print(clustering_samples_rewired)

        if settings.save_computations:

            pickle.dump(clustering_samples_add_random, open(output_directory_address + 'clustering_samples_' + 'add_random_'
                                                       + network_group + network_id + '.pkl', 'wb'))
            pickle.dump(clustering_samples_add_triad, open(output_directory_address + 'clustering_samples_' + 'add_triad_'
                                                      + network_group + network_id + '.pkl', 'wb'))

            pickle.dump(clustering_samples_rewired,
                        open(output_directory_address + 'clustering_samples_' + 'rewiring_'
                             + network_group + network_id + '.pkl', 'wb'))

        if settings.load_computations:
            clustering_samples_add_random = pickle.load(open(output_directory_address + 'clustering_samples_' + 'add_random_'
                                                             + network_group + network_id + '.pkl', 'rb'))

            clustering_samples_add_triad = pickle.load(open(output_directory_address + 'clustering_samples_' + 'add_triad_'
                                                       + network_group + network_id + '.pkl', 'rb'))

            clustering_samples_rewired = pickle.load(open(output_directory_address + 'clustering_samples_' + 'rewiring_'
                                                          + network_group + network_id + '.pkl', 'rb'))

            clustering_add_triad = np.mean(clustering_samples_add_triad)

            clustering_add_random = np.mean(clustering_samples_add_random)

            clustering_rewired = np.mean(clustering_samples_rewired)

            clustering_std_add_triad = np.std(clustering_samples_add_triad)

            clustering_std_add_random = np.std(clustering_samples_add_random)

            clustering_std_rewired = np.std(clustering_samples_rewired)

        clustering_original += [original_average_clustering]
        avg_clustering_add_random += [clustering_add_random]
        avg_clustering_add_triad += [clustering_add_triad]
        avg_clustering_rewired += [clustering_rewired]

        std_clustering_add_random += [clustering_std_add_random]
        std_clustering_add_triad += [clustering_std_add_triad]
        std_clustering_rewired += [clustering_std_rewired]

        if settings.data_dump:

            print('we are in data_dump mode')

            dataset = [[network_group, network_id, network_size,
                        number_edges, 'random_addition', percent_more_edges,
                        clustering_add_random],
                       [network_group, network_id, network_size,
                        number_edges, 'triad_addition', percent_more_edges,
                        clustering_add_triad],
                       [network_group, network_id, network_size,
                        number_edges, 'rewiring', rewiring_percentage,
                        clustering_rewired],
                       [network_group, network_id, network_size,
                        number_edges, 'none', rewiring_percentage,
                        original_average_clustering]]

            # df_add_random = pd.DataFrame(data=[[network_group, network_id, network_size,
            #                                     number_edges, 'random_addition', percent_more_edges,
            #                                     avg_clustering_add_random]],
            #                              columns=['network_group', 'network_id', 'network_size',
            #                                       'number_edges', 'intervention_type',
            #                                       'intervention_size',
            #                                       'avg_clustering'])
            #
            # df_add_triad = pd.DataFrame(data=[[network_group, network_id, network_size,
            #                                    number_edges, 'triad_addition', percent_more_edges,
            #                                    avg_clustering_add_triad]],
            #                             columns=['network_group', 'network_id', 'network_size',
            #                                      'number_edges', 'intervention_type',
            #                                      'intervention_size',
            #                                      'avg_clustering'])
            #
            # df_rewired = pd.DataFrame(data=[[network_group, network_id, network_size,
            #                                  number_edges, 'rewiring', rewiring_percentage,
            #                                  avg_clustering_rewired]],
            #                           columns=['network_group', 'network_id', 'network_size',
            #                                    'number_edges', 'intervention_type',
            #                                    'intervention_size',
            #                                    'avg_clustering'])
            #
            # df_original = pd.DataFrame(data=[[network_group, network_id, network_size,
            #                                  number_edges, 'none', rewiring_percentage,
            #                                  clustering_original]],
            #                           columns=['network_group', 'network_id', 'network_size',
            #                                    'number_edges', 'intervention_type',
            #                                    'intervention_size',
            #                                    'avg_clustering'])



            # df_add_random = pd.concat([df_common_part_add_random, df_sample_ids_add_random,
            #                                df_time_to_spreads_add_random],
            #                               axis=1)
            #
            # # print(new_df_add_random)
            #
            new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
                                                         'number_edges', 'intervention_type',
                                                         'intervention_size', 'avg_clustering'])
            #
            # # print(new_df)
            #
            # df_common_part_add_triad = pd.DataFrame(data=[[network_group, network_id, network_size,
            #                                                number_edges, 'triad_addition', percent_more_edges]]
            #                                              * len(speed_samples_add_triad),
            #                                         columns=['network_group', 'network_id', 'network_size',
            #                                                  'number_edges', 'intervention_type',
            #                                                  'intervention_size'])
            #
            # df_sample_ids_add_triad = pd.Series(list(range(len(speed_samples_add_triad))), name='sample_id')
            #
            # df_time_to_spreads_add_triad = pd.Series(speed_samples_add_triad, name='time_to_spread')
            #
            # new_df_add_triad = pd.concat([df_common_part_add_triad, df_sample_ids_add_triad,
            #                               df_time_to_spreads_add_triad],
            #                              axis=1)
            #
            # print(new_df_add_triad)

            print(new_df)
            # print(df_add_random)
            # print(df_original)
            # print(df_rewired)
            # print(df_add_triad)

            extended_frame = [df, new_df]#, df_add_triad, df_rewired, df_original]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
                drop=True)

            # print(df)

    if settings.do_plots:

        plt.figure()

        plt.hist([clustering_original,avg_clustering_add_random, avg_clustering_add_triad, avg_clustering_rewired],
                 label=['original',
                        str(percent_more_edges) + ' \% random edge addition',
                        str(percent_more_edges) + ' \% triad edge addition',
                        str(rewiring_percentage) + ' \% rewiring'])
        # plt.hist(speed_samples_rewired, label='rewired')

        plt.ylabel('Frequency')
        plt.xlabel('Average Clustering')
        plt.title('Average Clustering under Various Interventions')
        plt.legend()
        if settings.show_plots:
            plt.show()
            # if settings.layout == 'circular':
            #     positions = NX.circular_layout(G, scale=4)
            # elif settings.layout == 'spring':
            #     positions = NX.spring_layout(time_networks[time], scale=4)
            # NX.draw(time_networks[time],
            #         pos=positions,
            #         node_color=[time_networks[time].node[i]['state'] for i in time_networks[time].nodes()],
            #         with_labels=False,
            #         edge_color='c',
            #         cmap=PL.cm.YlOrRd,
            #         vmin=0,
            #         vmax=1)
        if settings.save_plots:
            plt.savefig(output_directory_address + 'avg_clustering_histogram_' + '.png')

    if settings.data_dump:
        df.to_csv(output_directory_address + network_group + 'clustering_data_dump.csv', index=False)#  , index=False

