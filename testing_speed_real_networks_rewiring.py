# Comparing the rate of contagion in the original and rewired network.
# loops over all files in
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

######################################################################
# # cai edgelists:
#
# network_group = 'cai_edgelist_'
#
# root_data_address = './data/cai-data/'
#
# edgelist_directory_address = root_data_address + 'edgelists/'
#
# output_directory_address = root_data_address + 'output/'
#
# DELIMITER = ' '

#####################################################################
# chami edgelists:

network_group = 'chami_edgelist_'

root_data_address = './data/chami-friendship-data/'

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

DELIMITER = ','

try:
    os.makedirs(output_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise

network_id_list = list(np.linspace(1,17,17))  # cannot do 152

network_id_list = [str(int(id)) for id in network_id_list]

rewiring_percentage = 5

if __name__ == '__main__':

    if data_dump:
        generating_new_dump = False
        try:
            df = pd.read_csv(output_directory_address + network_group + 'data_dump.csv')
            print('read_csv',df)
        except FileNotFoundError:

            df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                       'number_edges', 'intervention_type',
                                       'intervention_size', 'sample_id', 'time_to_spread'], dtype='float')
            print('New ' + network_group + 'data_dump file will be generated.')
            generating_new_dump = True

    for network_id in network_id_list:

        print(network_id)

        #  load in the network and extract preliminary data

        fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')


        G = NX.read_edgelist(fh,delimiter=DELIMITER)

        print('original size ',len(G.nodes()))

        if not NX.is_connected(G):
            G = max(NX.connected_component_subgraphs(G), key=len)
            print('largest connected component extracted with size ', len(G.nodes()))


        network_size = NX.number_of_nodes(G)

        number_edges = G.number_of_edges()

        if settings.do_computations:

            number_initial_seeds = 2

            params_original = {
                'network': G,
                'size': network_size,
                'add_edges': False,
                # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                'initialization_mode': 'fixed_number_initial_infection',
                'initial_infection_number': number_initial_seeds,
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': 1.0,
                'fixed_prob': 0.05,
                'theta': 2,
                'rewire': False,
                'rewiring_mode': 'random_random',
                'num_edges_for_random_random_rewiring': None,
                # rewire 15% of edges
            }

            dynamics_original = DeterministicLinear(params_original)
            speed_original,std_original, _, _, speed_samples_original = \
                dynamics_original.avg_speed_of_spread(
                    dataset_size=size_of_dataset,
                    cap=0.9,
                    mode='max')
            print(speed_original, std_original)
            print(speed_samples_original)

            print(type(speed_original))

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
                'num_edges_for_random_random_rewiring': 0.01*rewiring_percentage * G.number_of_edges(),
                # rewire 15% of edges
            }

            dynamics_rewired = DeterministicLinear(params_rewired)
            speed_rewired, std_rewired, _, _, speed_samples_rewired = \
                dynamics_rewired.avg_speed_of_spread(dataset_size=size_of_dataset,
                                                     cap=0.9,
                                                     mode='max')
            print(speed_rewired, std_rewired)
            print(speed_samples_rewired)

            print(NX.is_connected(G))

        if settings.save_computations:

            pickle.dump(speed_samples_rewired, open(output_directory_address + 'speed_samples_' + str(rewiring_percentage) +
                                                    '_percent_rewiring_' + network_group + network_id + '.pkl', 'wb'))
            pickle.dump(speed_samples_original, open(output_directory_address + 'speed_samples_original_'
                                                     + network_group + network_id + '.pkl', 'wb'))

        if settings.load_computations:

            speed_samples_rewired = pickle.load(open(output_directory_address + 'speed_samples_' + str(rewiring_percentage) +
                                                     '_percent_rewiring_' + network_group + network_id + '.pkl', 'rb'))
            speed_samples_original = pickle.load(open(output_directory_address + 'speed_samples_original_'
                                                      + network_group + network_id + '.pkl', 'rb'))

            speed_original = np.mean(speed_samples_original)

            speed_rewired = np.mean(speed_samples_rewired)

            std_original = np.std(speed_samples_original)

            std_rewired = np.std(speed_samples_rewired)

        if settings.do_plots:

            plt.figure()

            plt.hist([speed_samples_original, speed_samples_rewired], label=['original', 'rewired'])
            # plt.hist(speed_samples_rewired, label='rewired')

            plt.ylabel('Frequency')

            plt.xlabel('Time to Spread')

            plt.title('\centering The mean spread times are '
                      + str(Decimal(speed_original).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(std_original).quantize(TWOPLACES)) + ')'
                      + ' and '
                      + str(Decimal(speed_rewired).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(std_rewired).quantize(TWOPLACES)) + '),' +
                       '\\vspace{-10pt}  \\begin{center}  in the original and ' + str(rewiring_percentage)
                      + '\% rewired network. \\end{center}')

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
                plt.savefig(output_directory_address + 'speed_samples_histogram_'+str(rewiring_percentage)
                            + '_percent_rewiring_' + network_group + network_id + '.png')

        if settings.data_dump:
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

            df_common_part_original = pd.DataFrame(data=[[network_group, network_id, network_size,
                               number_edges, 'none', 0.0]]*len(speed_samples_original),
                                          columns=['network_group', 'network_id', 'network_size',
                                                   'number_edges', 'intervention_type',
                                                   'intervention_size'])

            df_sample_ids_original = pd.Series(list(range(len(speed_samples_original))),name='sample_id')

            df_time_to_spreads_original = pd.Series(speed_samples_original,name='time_to_spread')

            new_df_original = pd.concat([df_common_part_original, df_sample_ids_original, df_time_to_spreads_original], axis=1)

            print(new_df_original)

            # new_df = pd.DataFrame(data=dataset, columns=['network_group', 'network_id', 'network_size',
            #                                              'number_edges', 'intervention_type',
            #                                              'intervention_size', 'sample_id', 'time_to_spread'])

            # print(new_df)

            df_common_part_rewired = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                          number_edges, 'rewired', rewiring_percentage]] * len(speed_samples_rewired),
                                                   columns=['network_group', 'network_id', 'network_size',
                                                            'number_edges', 'intervention_type',
                                                            'intervention_size'])

            df_sample_ids_rewired = pd.Series(list(range(len(speed_samples_rewired))), name='sample_id')

            df_time_to_spreads_rewired = pd.Series(speed_samples_rewired, name='time_to_spread')

            new_df_rewired = pd.concat([df_common_part_rewired, df_sample_ids_rewired, df_time_to_spreads_rewired],
                                        axis=1)

            print(new_df_rewired)

            extended_frame = [df, new_df_original, new_df_rewired]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(drop=True)

            # df.drop_duplicates().reset_index(drop=True)



            # for ii in range(len(speed_samples_original)):
            #
            #     # new_df = pd.DataFrame(data=[[network_group, network_id, network_size, number_edges,
            #     #                               'none', 0.0, speed_samples_original[ii]]],
            #     #                       columns=['network_group', 'network_id', 'network_size',
            #     #                                'number_edges', 'intervention_type',
            #     #                                'intervention_size', 'time_to_spread'])
            #     # print(new_df)
            #     top_index = df.idxmax(axis=0, skipna=True)
            #     df.loc[top_index] = [network_group, network_id, network_size, number_edges, 'none', 0.0, speed_samples_original[ii]]
            #     # df.concat(new_df,ignore_index=True)
            # # df[network_group + network_id + '_rewired_' + str(rewiring_percentage) + '_percent'] = pd.Series(speed_samples_rewired)
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
