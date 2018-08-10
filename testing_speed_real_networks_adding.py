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


network_id_list = list(np.linspace(1,11,11))

network_id_list = [str(int(id)) for id in network_id_list]

percent_more_edges = 15

if __name__ == '__main__':

    if data_dump:
        try:
            df = pd.read_csv(output_directory_address + network_group + 'data_dump.csv')
        except FileNotFoundError:
            df = pd.DataFrame(dtype='float')
            print('New ' + network_group + 'data_dump file will be generated.')

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

            initial_seeds = 2

            params_add_random = {
                'network': G,
                'size': network_size,
                'add_edges': True,
                'edge_addition_mode': 'random',
                'number_of_edges_to_be_added': int(np.floor(0.01*percent_more_edges * G.number_of_edges())),
                # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                'initialization_mode': 'fixed_number_initial_infection',
                'initial_infection_number': 2,
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': 1.0,
                'fixed_prob': 0.05,
                'theta': 2,
                'rewire': False,
            }

            dynamics_add_random = DeterministicLinear(params_add_random)

            speed_add_random, std_add_random, _, _, speed_samples_add_random = \
                dynamics_add_random.avg_speed_of_spread(
                    dataset_size=size_of_dataset,
                    cap=0.9,
                    mode='max')

            print(speed_add_random, std_add_random)

            print(speed_samples_add_random)

            params_add_triad = {
                'network': G,
                'size': network_size,
                'add_edges': True,
                'edge_addition_mode': 'triadic_closures',
                'number_of_edges_to_be_added': int(np.floor(0.01 * percent_more_edges * G.number_of_edges())),
                # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                'initialization_mode': 'fixed_number_initial_infection',
                'initial_infection_number': 2,
                'delta': 0.0000000000000001,
                'fixed_prob_high': 1.0,
                'fixed_prob': 0.05,
                'theta': 2,
                'rewire': False,
                # rewire 10% of edges
            }

            dynamics_add_triad = DeterministicLinear(params_add_triad)

            speed_add_triad, std_add_triad, _, _, speed_samples_add_triad = \
                dynamics_add_triad.avg_speed_of_spread(
                    dataset_size=size_of_dataset,
                    cap=0.9,
                    mode='max')

            print(speed_add_triad, std_add_triad)

            print(speed_samples_add_triad)

            print(NX.is_connected(G))

        if settings.save_computations:

            pickle.dump(speed_samples_add_random, open(output_directory_address + 'speed_samples_' + 'add_random_'
                                                       + network_group + network_id + '.pkl', 'wb'))
            pickle.dump(speed_samples_add_triad, open(output_directory_address + 'speed_samples_' + 'add_triad_'
                                                      + network_group + network_id + '.pkl', 'wb'))


        if settings.load_computations:
            speed_samples_add_random = pickle.load(open(output_directory_address + 'speed_samples_' + 'add_random_'
                                                        + network_group + network_id + '.pkl', 'rb'))
            speed_samples_add_triad = pickle.load(open(output_directory_address + 'speed_samples_' + 'add_triad_'
                                                       + network_group + network_id + '.pkl', 'rb'))

            speed_add_triad = np.mean(speed_samples_add_triad)

            speed_add_random = np.mean(speed_samples_add_random)

            std_add_triad = np.std(speed_samples_add_triad)

            std_add_random = np.std(speed_samples_add_random)

        if settings.do_plots:

            plt.figure()

            plt.hist([speed_samples_add_random, speed_samples_add_triad], label=['random', 'triads'])
            # plt.hist(speed_samples_rewired, label='rewired')

            plt.ylabel('Frequency')
            plt.xlabel('Time to Spread')
            plt.title('\centering The mean spread times are '
                      + str(Decimal(speed_add_random).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(std_add_random).quantize(TWOPLACES)) + ')'
                      + ' and '
                      + str(Decimal(speed_add_triad).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(std_add_triad).quantize(TWOPLACES)) + '),' +
                       '\\vspace{-10pt}  \\begin{center}  in the two networks with ' + str(percent_more_edges)
                      + '\% additional random or triad closing edges. \\end{center}')
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
                plt.savefig(output_directory_address + 'speed_samples_histogram_'
                            + str(percent_more_edges) + '_edge_additions_'
                            + network_group + network_id + '.png')
        if settings.data_dump:
            # print('we are in data_dump mode')
            # df[network_group + network_id + '_add_random_' + str(percent_more_edges) + '_percent'] = pd.Series(speed_samples_add_random)
            # df[network_group + network_id + '_add_triad_' + str(percent_more_edges) + '_percent'] = pd.Series(speed_samples_add_triad)
            # print(df)
            print('we are in data_dump mode')

            # speed_samples_original = [str(samples) for samples in speed_samples_original]
            #
            # print(speed_samples_original)
            #
            # speed_samples_rewired = [str(samples) for samples in speed_samples_rewired]
            #
            # print(speed_samples_rewired)

            # data = [speed_samples_original, speed_samples_rewired]
            # print(df)
            # df.loc[0] = [network_group, network_id, network_size, number_edges, 'none', 0.0,
            #                      speed_samples_original[0]]

            # dataset = [[network_group, network_id, network_size, number_edges,
            #             'none', 0.0, int(ii), speed_samples_original[ii]]
            #            for ii in range(len(speed_samples_original))]

            df_common_part_add_random = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                          number_edges, 'random_addition', percent_more_edges]]
                                                          * len(speed_samples_add_random),
                                                   columns=['network_group', 'network_id', 'network_size',
                                                            'number_edges', 'intervention_type',
                                                            'intervention_size'])

            df_sample_ids_add_random = pd.Series(list(range(len(speed_samples_add_random))), name='sample_id')

            df_time_to_spreads_add_random = pd.Series(speed_samples_add_random, name='time_to_spread')

            new_df_add_random = pd.concat([df_common_part_add_random, df_sample_ids_add_random,
                                           df_time_to_spreads_add_random],
                                          axis=1)

            print(new_df_add_random)

            # new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
            #                                              'number_edges', 'intervention_type',
            #                                              'intervention_size', 'sample_id', 'time_to_spread'])

            # print(new_df)

            df_common_part_add_triad = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                           number_edges, 'triad_addition', percent_more_edges]]
                                                         *len(speed_samples_add_triad),
                                                    columns=['network_group', 'network_id', 'network_size',
                                                             'number_edges', 'intervention_type',
                                                             'intervention_size'])

            df_sample_ids_add_triad = pd.Series(list(range(len(speed_samples_add_triad))), name='sample_id')

            df_time_to_spreads_add_triad = pd.Series(speed_samples_add_triad, name='time_to_spread')

            new_df_add_triad = pd.concat([df_common_part_add_triad, df_sample_ids_add_triad,
                                          df_time_to_spreads_add_triad],
                                         axis=1)

            print(new_df_add_triad)

            extended_frame = [df, new_df_add_random, new_df_add_triad]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
                drop=True)

            print(df)

    if settings.data_dump:
        df.to_csv(output_directory_address + network_group + 'data_dump.csv', index=False)#  , index=False


            # with open(output_directory_address + 'names.csv', 'w') as csvfile:
            #     fieldnames = ['first_name', 'last_name']
            #
            #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            #
            #     writer.writeheader()
            #     for ii in range(size_of_dataset):
            #         writer.writerow({'first_name': speed_samples_original[ii], 'last_name': speed_samples_rewired[ii]})
            #     # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
            #     # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
