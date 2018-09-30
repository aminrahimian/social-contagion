# goes over pickled spreading times  and puts their into the CSV file: network_group + 'spreading_data_dump.csv'

from settings import *


rewiring_percentage_list = [5, 10, 15, 20, 25]

percent_more_edges_list = [5, 10, 15, 20, 25]

include_original_networks = False
include_rewiring_networks = False
include_addition_networks = True

update_existing_dump = True

if __name__ == "__main__":

    assert data_dump, "we should be in data_dump mode!"

    assert load_computations, "we should be in load_computations mode to dump data!"

    # check for the existing network_group + 'spreading_data_dump.csv' file

    generating_new_dump = False

    try:
        df = pd.read_csv(output_directory_address + network_group + 'spreading_data_dump.csv')
        print('read_csv', df)
    except FileNotFoundError:

        df = pd.DataFrame(columns=['network_group', 'network_id', 'network_size',
                                   'intervention_type',
                                   'intervention_size', 'sample_id', 'time_to_spread', 'model'], dtype='float')
        print('New ' + network_group + 'spreading_data_dump file will be generated.')
        generating_new_dump = True

    if update_existing_dump:
        assert not generating_new_dump, "we should not be generating a new spreading_data_dump file."

    for network_id in network_id_list:

        print('load/dump speed_samples_original_' + network_group + network_id + model_id)

        #  load in the network and extract preliminary data

        fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

        G = NX.read_edgelist(fh, delimiter=DELIMITER)

        print('original size ', len(G.nodes()))

        #  get the largest connected component:
        if not NX.is_connected(G):
            G = max(NX.connected_component_subgraphs(G), key=len)
            print('largest connected component extracted with size ', len(G.nodes()))

        network_size = NX.number_of_nodes(G)

        if include_original_networks:

            speed_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                      + 'speed_samples_original_'
                                                      + network_group + network_id
                                                      + model_id + '.pkl', 'rb'))

            speed_original = np.mean(speed_samples_original)

            std_original = np.std(speed_samples_original)

            # dump original:

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

        if include_rewiring_networks:

            # dump rewired network spreading times:

            for rewiring_percentage in rewiring_percentage_list:

                # load data:

                print('load/dump' + str(rewiring_percentage) +
                      '_percent_rewiring_' + network_group + network_id + model_id)

                speed_samples_rewired = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                         + str(rewiring_percentage) +
                                                         '_percent_rewiring_' + network_group + network_id
                                                         + model_id + '.pkl', 'rb'))

                speed_rewired = np.mean(speed_samples_rewired)

                std_rewired = np.std(speed_samples_rewired)

                # dump the loaded data:

                df_common_part_rewired = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                             'rewired', rewiring_percentage, MODEL]]
                                                           * len(speed_samples_rewired),
                                                      columns=['network_group', 'network_id', 'network_size',
                                                               'intervention_type',
                                                               'intervention_size', 'model'])

                df_sample_ids_rewired = pd.Series(list(range(len(speed_samples_rewired))), name='sample_id')

                df_time_to_spreads_rewired = pd.Series(speed_samples_rewired, name='time_to_spread')

                new_df_rewired = pd.concat([df_common_part_rewired, df_sample_ids_rewired, df_time_to_spreads_rewired],
                                           axis=1)

                print(new_df_rewired)

                extended_frame = [df, new_df_rewired]

                df = pd.concat(extended_frame, ignore_index=True,
                               verify_integrity=False)  # .drop_duplicates().reset_index(drop=True)

                print(df)

        if include_addition_networks:

            # dump the edge added network spreading times:

            for percent_more_edges in percent_more_edges_list:

                # load data:
                print('load/dump' + str(percent_more_edges) + '_percent_' + 'add_random_'
                      + network_group + network_id + model_id)
                print('load/dump' + str(percent_more_edges) + '_percent_' + 'add_triad_'
                      + network_group + network_id + model_id)

                speed_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                            + str(percent_more_edges) + '_percent_' + 'add_random_'
                                                            + network_group + network_id
                                                            + model_id + '.pkl', 'rb'))
                speed_samples_add_triad = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                           + str(percent_more_edges) + '_percent_' + 'add_triad_'
                                                           + network_group + network_id
                                                           + model_id + '.pkl', 'rb'))

                speed_add_triad = np.mean(speed_samples_add_triad)

                speed_add_random = np.mean(speed_samples_add_random)

                std_add_triad = np.std(speed_samples_add_triad)

                std_add_random = np.std(speed_samples_add_random)

                # dump edge addition networks spreading data (after loading):

                df_common_part_add_random = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                                'random_addition', percent_more_edges,
                                                                MODEL]] * len(speed_samples_add_random),
                                                         columns=['network_group', 'network_id', 'network_size',
                                                                  'intervention_type',
                                                                  'intervention_size', 'model'])

                df_sample_ids_add_random = pd.Series(list(range(len(speed_samples_add_random))), name='sample_id')

                df_time_to_spreads_add_random = pd.Series(speed_samples_add_random, name='time_to_spread')

                new_df_add_random = pd.concat([df_common_part_add_random, df_sample_ids_add_random,
                                               df_time_to_spreads_add_random],
                                              axis=1)

                df_common_part_add_triad = pd.DataFrame(data=[[network_group, network_id, network_size,
                                                               'triad_addition', percent_more_edges,
                                                               MODEL]] * len(speed_samples_add_triad),
                                                        columns=['network_group', 'network_id', 'network_size',
                                                                 'intervention_type',
                                                                 'intervention_size', 'model'])

                df_sample_ids_add_triad = pd.Series(list(range(len(speed_samples_add_triad))), name='sample_id')

                df_time_to_spreads_add_triad = pd.Series(speed_samples_add_triad, name='time_to_spread')

                new_df_add_triad = pd.concat([df_common_part_add_triad, df_sample_ids_add_triad,
                                              df_time_to_spreads_add_triad],
                                             axis=1)

                print(new_df_add_triad)

                extended_frame = [df, new_df_add_random, new_df_add_triad]

                df = pd.concat(extended_frame, ignore_index=True,
                               verify_integrity=False)  # .drop_duplicates().reset_index(drop=True)

        df.to_csv(output_directory_address + network_group + 'spreading_data_dump.csv', index=False)  # , index=False


