# Comparing the rate of contagion in the original and rewired network.
# loops over all files in
# uses maslov_sneppen_rewiring(num_steps = np.floor(0.1 * self.params['network'].number_of_edges()))
# to rewire the network
# uses avg_speed_of_spread(dataset_size=1000,cap=0.9, mode='max') to measure the rate of spread

from settings import *
import settings

# assert settings.data_dump, "we should be in data_dump mode!"

from models import *

size_of_dataset = 200

# network_id_list = list(np.linspace(1, 17, TOP_ID))  # cannot do 152
#
# network_id_list = [str(int(id)) for id in network_id_list]

rewiring_percentage_list = [5, 10, 15, 20, 25]
loop_mode = (len(rewiring_percentage_list) > 1)
print(loop_mode)

def check_type(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj

if __name__ == '__main__':

    if data_dump:
        generating_new_dump = False
        try:
            df = pd.read_csv(output_directory_address + network_group + 'spreading_data_dump.csv')
            print('read_csv',df)
        except FileNotFoundError:

            df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                       'intervention_type',
                                       'intervention_size', 'sample_id', 'time_to_spread', 'model'], dtype='float')
            print('New ' + network_group + 'spreading_data_dump file will be generated.')
            generating_new_dump = True

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

        # number_edges = G.number_of_edges()

        if settings.do_computations:

            params_original = {
                'network': G,
                'size': network_size,
                'add_edges': False,
                'initialization_mode': 'fixed_number_initial_infection',
                'initial_infection_number': number_initial_seeds,
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': fixed_prob_high,
                'fixed_prob': fixed_prob_low,
                'alpha': alpha,
                'gamma': gamma,
                'theta': 2,
                'rewire': False,
                'rewiring_mode': 'random_random',
                'num_edges_for_random_random_rewiring': None,
            }

            dynamics_original = DeterministicLinear(params_original)
            speed_original, std_original, _, _, speed_samples_original = \
                dynamics_original.avg_speed_of_spread(
                    dataset_size=size_of_dataset,
                    cap=0.9,
                    mode='max')
            print(speed_original, std_original)
            print(speed_samples_original)

            print(type(speed_original))

        if settings.save_computations:

            pickle.dump(speed_samples_original, open(pickled_samples_directory_address + 'speed_samples_original_'
                                                     + network_group + network_id
                                                     + model_id + '.pkl', 'wb'))

        if settings.load_computations:

            speed_samples_original = pickle.load(open(pickled_samples_directory_address + 'speed_samples_original_'
                                                      + network_group + network_id
                                                      + model_id + '.pkl', 'rb'))

            speed_original = np.mean(speed_samples_original)

            std_original = np.std(speed_samples_original)

        if settings.data_dump:
            print('we are in data_dump mode')

            df_common_part_original = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                          'none', 0.0, MODEL]] * len(speed_samples_original),
                                                   columns=['network_group', 'network_id', 'network_size',
                                                            'intervention_type',
                                                            'intervention_size','model'])

            df_sample_ids_original = pd.Series(list(range(len(speed_samples_original))), name='sample_id')

            df_time_to_spreads_original = pd.Series(speed_samples_original, name='time_to_spread')

            new_df_original = pd.concat([df_common_part_original, df_sample_ids_original, df_time_to_spreads_original],
                                        axis=1)

            print(new_df_original)

            extended_frame = [df, new_df_original]

            df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False).drop_duplicates().reset_index(
                drop=True)

            print(df)

        for rewiring_percentage in rewiring_percentage_list:

            if settings.do_computations:

                params_rewired = {
                    'network': G,
                    'size': network_size,
                    'add_edges': False,
                    # 'initial_states': [1] * initial_seeds + [0] * (network_size - initial_seeds),
                    'initialization_mode': 'fixed_number_initial_infection',
                    'initial_infection_number': number_initial_seeds,
                    'delta': 0.0000000000000001,
                    'fixed_prob_high': fixed_prob_high,
                    'fixed_prob': fixed_prob_low,
                    'alpha': alpha,
                    'gamma': gamma,
                    'theta': 2,
                    'rewire': True,
                    'rewiring_mode': 'random_random',
                    'num_edges_for_random_random_rewiring': 0.01*rewiring_percentage * G.number_of_edges(),
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

                pickle.dump(speed_samples_rewired, open(pickled_samples_directory_address + 'speed_samples_'
                                                        + str(rewiring_percentage) +
                                                        '_percent_rewiring_' + network_group + network_id
                                                        + model_id + '.pkl', 'wb'))

            if settings.load_computations:

                speed_samples_rewired = pickle.load(open(pickled_samples_directory_address + 'speed_samples_'
                                                         + str(rewiring_percentage) +
                                                         '_percent_rewiring_' + network_group + network_id
                                                         + model_id + '.pkl', 'rb'))

                speed_rewired = np.mean(speed_samples_rewired)

                std_rewired = np.std(speed_samples_rewired)

            if settings.do_plots:

                plt.figure()

                plt.hist([speed_samples_original, speed_samples_rewired], label=['original', 'rewired'])

                plt.ylabel('Frequency')

                plt.xlabel('Time to Spread')

                plt.title('\centering The mean spread times are '
                          + str(Decimal(check_type(speed_original)).quantize(TWOPLACES))
                          + '(SD=' + str(Decimal(check_type(std_original)).quantize(TWOPLACES)) + ')'
                          + ' and '
                          + str(Decimal(check_type(speed_rewired)).quantize(TWOPLACES))
                          + '(SD=' + str(Decimal(check_type(std_rewired)).quantize(TWOPLACES)) + '),'
                          + '\\vspace{-10pt}  \\begin{center}  in the original and ' + str(rewiring_percentage)
                          + '\% rewired network. \\end{center}')

                plt.legend()

                if settings.show_plots:
                    plt.show()

                if settings.save_plots:
                    plt.savefig(output_directory_address + 'speed_samples_histogram_'+str(rewiring_percentage)
                                + '_percent_rewiring_' + network_group + network_id
                                + model_id + '.png')

            if settings.data_dump:
                print('we are in data_dump mode')

                df_common_part_rewired = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                             'rewired', rewiring_percentage, MODEL]]
                                                           * len(speed_samples_rewired),
                                                      columns=['network_group', 'network_id', 'network_size',
                                                               'intervention_type',
                                                               'intervention_size','model'])

                df_sample_ids_rewired = pd.Series(list(range(len(speed_samples_rewired))), name='sample_id')

                df_time_to_spreads_rewired = pd.Series(speed_samples_rewired, name='time_to_spread')

                new_df_rewired = pd.concat([df_common_part_rewired, df_sample_ids_rewired, df_time_to_spreads_rewired],
                                            axis=1)

                print(new_df_rewired)

                extended_frame = [df, new_df_rewired]

                df = pd.concat(extended_frame, ignore_index=True, verify_integrity=False)#.drop_duplicates().reset_index(drop=True)

                print(df)

    if settings.data_dump:
        df.to_csv(output_directory_address + network_group + 'spreading_data_dump.csv', index=False)#  , index=False
