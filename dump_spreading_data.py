# goes over pickled spreading times  and puts their into the CSV file: network_group + 'spreading_data_dump.csv'

from settings import *


rewiring_percentage_list = [10] #[5, 10, 15, 20, 25]  # [5, 10, 15, 20, 25]

percent_more_edges_list = [10] #[5, 10, 15, 20, 25]  # [5, 10, 15, 20, 25]


include_original_networks = True
include_rewiring_networks = True
include_addition_networks = True
include_add_rand_networks = False

assert not (include_add_rand_networks and include_addition_networks), \
    "include_add_rand_networks and include_addition_networks cannot be both True"

include_spread_size = True

if not include_spread_size:

    tracked_properties = ['network_group',
                          'network_id',
                          'network_size',
                          'theta_distribution',
                          'intervention_type',
                          'intervention_size',
                          'sample_id',
                          'time_to_spread',
                          'model']
else:

    tracked_properties = ['network_group',
                          'network_id',
                          'network_size',
                          'theta_distribution',
                          'intervention_type',
                          'intervention_size',
                          'sample_id',
                          'time_to_spread',
                          'model',
                          'size_of_spread']


assert include_spread_size, "data dump without spread size is not supported!"

update_existing_dump = False


if __name__ == "__main__":

    assert data_dump, "we should be in data_dump mode!"

    assert load_computations, "we should be in load_computations mode to dump data!"

    # check for the existing network_group + 'spreading_data_dump.csv' file

    generating_new_dump = True

    try:
        df = pd.read_csv(output_directory_address + network_group + 'spreading_data_dump.csv')
        print('read_csv', df)
    except IOError:

        df = pd.DataFrame(columns=tracked_properties, dtype='float')
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

        for theta in theta_list:

            if include_original_networks:

                speed_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                      + 'speed_samples_original_'
                                                      + network_group + network_id
                                                      + model_id + '_' + str(theta) + '.pkl', 'rb'))

                spread_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                       + 'infection_size_original_'
                                                       + network_group + network_id
                                                       + model_id + '_' + str(theta) + '.pkl', 'rb'))

                ##Load Fractional series

                fractional_evolution_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                           + 'fractional_evolution_original_'
                                                           + network_group + network_id
                                                           + model_id + '_' + str(theta) + '.pkl', 'rb'))


                speed_original = np.mean(speed_samples_original)

                #Duplicate: spread_original = np.mean(spread_samples_original)

                std_original = np.std(speed_samples_original)

                spread_std_original = np.std(spread_samples_original)

                # dump original:



                df_common_part_original = pd.DataFrame(data=[[network_group, network_id, network_size, str(theta),
                                                          'none', 0.0, MODEL]] *len(speed_samples_original),
                                                   columns=['network_group',
                                                            'network_id',
                                                            'network_size',
                                                            'theta_distribution',
                                                            'intervention_type',
                                                            'intervention_size',
                                                            'model'])

                df_sample_ids_original = pd.Series(list(range(len(speed_samples_original))), name='sample_id')

                df_time_to_spreads_original = pd.Series(speed_samples_original, name='time_to_spread')

                df_size_of_spreads_original = pd.Series(spread_samples_original, name='size_of_spread')

                data_fractional_evolution_original = pd.Series(fractional_evolution_original, name='fractional_evolution')

                tuples_fractional = list(tuple(sub) for sub in data_fractional_evolution_original)

                df_fractional_evolution_original = pd.Series(tuples_fractional, name='fractional_evolution')

                new_df_original = pd.concat([df_common_part_original,
                                             df_sample_ids_original,
                                             df_time_to_spreads_original,
                                             df_size_of_spreads_original,
                                             df_fractional_evolution_original],
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
                          '_percent_rewiring_' + network_group + network_id + model_id + '_' + str(theta))

                    speed_samples_rewired = pickle.load(open(spreading_pickled_samples_directory_address
                                                             + 'speed_samples_'
                                                             + str(rewiring_percentage)
                                                             + '_percent_rewiring_'
                                                             + network_group
                                                             + network_id
                                                             + model_id + '_' + str(theta)
                                                             + '.pkl', 'rb'))

                    spread_samples_rewired = pickle.load(open(spreading_pickled_samples_directory_address
                                                              + 'infection_size_samples_'
                                                              + str(rewiring_percentage)
                                                              + '_percent_rewiring_'
                                                              + network_group
                                                              + network_id
                                                              + model_id + '_' + str(theta)
                                                              + '.pkl', 'rb'))

                    #Load Fractional series:

                    fractional_evolution_rewired = pickle.load(open(spreading_pickled_samples_directory_address
                                                              + 'fractional_evolution_samples_'
                                                              + str(rewiring_percentage)
                                                              + '_percent_rewiring_'
                                                              + network_group
                                                              + network_id
                                                              + model_id + '_' + str(theta)
                                                              + '.pkl', 'rb'))



                    speed_rewired = np.mean(speed_samples_rewired)

                    spread_rewired = np.mean(spread_samples_rewired)

                    std_rewired = np.std(speed_samples_rewired)

                    spread_std_rewired = np.std(spread_samples_rewired)



                    # dump the loaded data:

                    df_common_part_rewired = pd.DataFrame(data=[[network_group, network_id, network_size, str(theta),
                                                                 'rewired', rewiring_percentage, MODEL]]
                                                               *len(speed_samples_rewired),
                                                          columns=['network_group',
                                                                   'network_id',
                                                                   'network_size',
                                                                   'theta_distribution',
                                                                   'intervention_type',
                                                                   'intervention_size',
                                                                   'model'])

                    df_sample_ids_rewired = pd.Series(list(range(len(speed_samples_rewired))), name='sample_id')

                    df_time_to_spreads_rewired = pd.Series(speed_samples_rewired, name='time_to_spread')

                    df_size_of_spreads_rewired = pd.Series(spread_samples_rewired, name='size_of_spread')

                    data_fractional_evolution_rewired = pd.Series(fractional_evolution_rewired)

                    tuples_fractional = list(tuple(sub) for sub in data_fractional_evolution_rewired)

                    df_fractional_evolution_rewired = pd.Series(tuples_fractional, name='fractional_evolution')


                    new_df_rewired = pd.concat([df_common_part_rewired,
                                                df_sample_ids_rewired,
                                                df_time_to_spreads_rewired,
                                                df_size_of_spreads_rewired,
                                                df_fractional_evolution_rewired],
                                               axis=1)

                    print(new_df_rewired)

                    extended_frame = [df, new_df_rewired]

                    df = pd.concat(extended_frame,
                                   ignore_index=True,
                                   verify_integrity=False)  # .drop_duplicates().reset_index(drop=True)

                    print(df)

            if include_addition_networks:

                # dump the edge added network spreading times:

                for percent_more_edges in percent_more_edges_list:

                    # load data:

                    print('load/dump' + str(percent_more_edges) + '_percent_' + 'add_random_'
                          + network_group + network_id + model_id + '_' + str(theta))

                    print('load/dump' + str(percent_more_edges) + '_percent_' + 'add_triad_'
                          + network_group + network_id + model_id + '_' + str(theta))

                    speed_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                + 'speed_samples_'
                                                                + str(percent_more_edges)
                                                                + '_percent_'
                                                                + 'add_random_'
                                                                + network_group
                                                                + network_id
                                                                + model_id + '_' + str(theta)
                                                                + '.pkl', 'rb'))

                    speed_samples_add_triad = pickle.load(open(spreading_pickled_samples_directory_address
                                                               + 'speed_samples_'
                                                               + str(percent_more_edges)
                                                               + '_percent_'
                                                               + 'add_triad_'
                                                               + network_group
                                                               + network_id
                                                               + model_id + '_' + str(theta)
                                                               + '.pkl', 'rb'))

                    spread_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                 + 'infection_size_samples_'
                                                                 + str(percent_more_edges)
                                                                 + '_percent_'
                                                                 + 'add_random_'
                                                                 + network_group
                                                                 + network_id
                                                                 + model_id + '_' + str(theta)
                                                                 + '.pkl', 'rb'))

                    spread_samples_add_triad = pickle.load(open(spreading_pickled_samples_directory_address
                                                                + 'infection_size_samples_'
                                                                + str(percent_more_edges)
                                                                + '_percent_'
                                                                + 'add_triad_'
                                                                + network_group
                                                                + network_id
                                                                + model_id + '_' + str(theta)
                                                                + '.pkl', 'rb'))

                    fractional_evolution_add_triad = pickle.load(open(spreading_pickled_samples_directory_address
                                                                + 'fractional_evolution_samples_'
                                                                + str(percent_more_edges)
                                                                + '_percent_'
                                                                + 'add_triad_'
                                                                + network_group
                                                                + network_id
                                                                + model_id + '_' + str(theta)
                                                                + '.pkl', 'rb'))

                    fractional_evolution_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                      + 'fractional_evolution_samples_'
                                                                      + str(percent_more_edges)
                                                                      + '_percent_'
                                                                      + 'add_random_'
                                                                      + network_group
                                                                      + network_id
                                                                      + model_id + '_' + str(theta)
                                                                      + '.pkl', 'rb'))




                    speed_add_triad = np.mean(speed_samples_add_triad)

                    speed_add_random = np.mean(speed_samples_add_random)

                    spread_add_triad = np.mean(spread_samples_add_triad)

                    spread_add_random = np.mean(spread_samples_add_random)

                    std_add_triad = np.std(speed_samples_add_triad)

                    std_add_random = np.std(speed_samples_add_random)

                    spread_std_add_triad = np.std(spread_samples_add_triad)

                    spread_std_add_random = np.std(spread_samples_add_random)

                    # dump edge addition networks spreading data (after loading):

                    df_common_part_add_random = pd.DataFrame(data=[[network_group, network_id, network_size, str(theta),
                                                                    'random_addition', percent_more_edges,
                                                                    MODEL]] * len(speed_samples_add_random),
                                                             columns=['network_group', 'network_id', 'network_size',
                                                                      'theta_distribution', 'intervention_type',
                                                                      'intervention_size', 'model'])

                    df_sample_ids_add_random = pd.Series(list(range(len(speed_samples_add_random))), name='sample_id')

                    df_time_to_spreads_add_random = pd.Series(speed_samples_add_random, name='time_to_spread')

                    df_size_of_spreads_add_random = pd.Series(spread_samples_add_random, name='size_of_spread')

                    data_fractional_evolution_add_random = pd.Series(fractional_evolution_add_random)

                    tuples_fractional = list(tuple(sub) for sub in data_fractional_evolution_add_random)

                    df_fractional_evolution_add_random= pd.Series(tuples_fractional, name='fractional_evolution')

                    #Transform list of list in fractional series to pass correctly to dataframe



                    new_df_add_random = pd.concat([df_common_part_add_random,
                                                   df_sample_ids_add_random,
                                                   df_time_to_spreads_add_random,
                                                   df_size_of_spreads_add_random,
                                                   df_fractional_evolution_add_random],
                                                  axis=1)

                    df_common_part_add_triad = pd.DataFrame(data=[[network_group,
                                                                   network_id,
                                                                   network_size,
                                                                   str(theta),
                                                                   'triad_addition',
                                                                   percent_more_edges,
                                                                   MODEL]] * len(speed_samples_add_triad),
                                                            columns=['network_group',
                                                                     'network_id',
                                                                     'network_size',
                                                                     'theta_distribution',
                                                                     'intervention_type',
                                                                     'intervention_size',
                                                                     'model'])

                    df_sample_ids_add_triad = pd.Series(list(range(len(speed_samples_add_triad))), name='sample_id')

                    df_time_to_spreads_add_triad = pd.Series(speed_samples_add_triad, name='time_to_spread')

                    df_size_of_spreads_add_triad = pd.Series(spread_samples_add_triad, name='size_of_spread')

                    data_fractional_evolution_add_triad = pd.Series(fractional_evolution_add_triad)

                    tuples_fractional = list(tuple(sub) for sub in data_fractional_evolution_add_triad)

                    df_fractional_evolution_add_triad = pd.Series(tuples_fractional, name='fractional_evolution')

                    new_df_add_triad = pd.concat([df_common_part_add_triad,
                                                  df_sample_ids_add_triad,
                                                  df_time_to_spreads_add_triad,
                                                  df_size_of_spreads_add_triad,
                                                  df_fractional_evolution_add_triad],
                                                 axis=1)

                    print(new_df_add_triad)

                    extended_frame = [df, new_df_add_random, new_df_add_triad]

                    df = pd.concat(extended_frame, ignore_index=True,
                                   verify_integrity=False)  # .drop_duplicates().reset_index(drop=True)

            if include_add_rand_networks:

                # dump the edge added network spreading times:

                for percent_more_edges in percent_more_edges_list:

                    # load data:

                    print('load/dump' + str(percent_more_edges) + '_percent_' + 'add_random_'
                          + network_group + network_id + model_id + '_' + str(theta))

                    speed_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                + 'speed_samples_'
                                                                + str(percent_more_edges)
                                                                + '_percent_'
                                                                + 'add_random_'
                                                                + network_group
                                                                + network_id
                                                                + model_id + '_' + str(theta)
                                                                + '.pkl', 'rb'))

                    spread_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                 + 'infection_size_samples_'
                                                                 + str(percent_more_edges)
                                                                 + '_percent_'
                                                                 + 'add_random_'
                                                                 + network_group
                                                                 + network_id
                                                                 + model_id + '_' + str(theta)
                                                                 + '.pkl', 'rb'))

                    fractional_evolution_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address
                                                                 + 'fractional_evolution_samples_'
                                                                 + str(percent_more_edges)
                                                                 + '_percent_'
                                                                 + 'add_random_'
                                                                 + network_group
                                                                 + network_id
                                                                 + model_id + '_' + str(theta)
                                                                 + '.pkl', 'rb'))

                    speed_add_random = np.mean(speed_samples_add_random)

                    spread_add_random = np.mean(spread_samples_add_random)

                    std_add_random = np.std(speed_samples_add_random)

                    spread_std_add_random = np.std(spread_samples_add_random)

                    # dump edge addition networks spreading data (after loading):

                    df_common_part_add_random = pd.DataFrame(data=[[network_group, network_id, network_size, str(theta),
                                                                    'random_addition', percent_more_edges,
                                                                    MODEL]] * len(speed_samples_add_random),
                                                             columns=['network_group', 'network_id', 'network_size',
                                                                      'theta_distribution',
                                                                      'intervention_type',
                                                                      'intervention_size', 'model'])

                    df_sample_ids_add_random = pd.Series(list(range(len(speed_samples_add_random))), name='sample_id')

                    df_time_to_spreads_add_random = pd.Series(speed_samples_dadd_random, name='time_to_spread')

                    df_size_of_spreads_add_random = pd.Series(spread_samples_add_random, name='size_of_spread')

                    data_fractional_evolution_add_random = pd.Series(fractional_evolution_samples_add_random)

                    tuples_fractional = list(tuple(sub) for sub in data_fractional_evolution_add_random)

                    df_fractional_evolution_add_random = pd.Series(tuples_fractional, name='fractional_evolution')


                    new_df_add_random = pd.concat([df_common_part_add_random,
                                                   df_sample_ids_add_random,
                                                   df_time_to_spreads_add_random,
                                                   df_size_of_spreads_add_random,
                                                   df_fractional_evolution_add_random],
                                                  axis=1)

                    print(new_df_add_random)

                    extended_frame = [df, new_df_add_random]

                    df = pd.concat(extended_frame, ignore_index=True,
                                   verify_integrity=False)  # .drop_duplicates().reset_index(drop=True)

            df.to_csv(output_directory_address + network_group + 'spreading_data_dump.csv', index=False)  # , index=False

    for theta in theta_list:
    #splitting the original dump file (which included all thetas) into separate files for each theta
        df = pd.read_csv(output_directory_address + network_group + 'spreading_data_dump.csv')

        #Filter the desired rows from our original dump file
        criterion = df['theta_distribution'] == str(theta)

        df_new = df[criterion]

        df_new.to_csv(output_directory_address + network_group + 'spreading_data_dump' + '_' + str(theta) + '.csv')