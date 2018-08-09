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

prefix = 'cai_edgelist_'

root_data_address = './data/cai-data/'

edgelist_directory_address = root_data_address + 'edgelists/'

output_directory_address = root_data_address + 'output/'

try:
    os.makedirs(output_directory_address)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise


ID_list = list(np.linspace(1,11,11))

ID_list = [str(int(ID)) for ID in ID_list]

percent_more_edges = 15

if __name__ == '__main__':

    if data_dump:
        try:
            df = pd.read_csv(output_directory_address + prefix + 'data_dump.csv')
        except FileNotFoundError:
            df = pd.DataFrame(dtype='float')
            print('New ' + prefix + 'data_dump file will be generated.')

    for ID in ID_list:

        print(ID)

        if settings.do_computations:

            fh = open(edgelist_directory_address + prefix + ID + '.txt', 'rb')

            G = NX.read_edgelist(fh)

            print(NX.is_connected(G))

            network_size = NX.number_of_nodes(G)

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
                                                       + prefix + ID + '.pkl', 'wb'))
            pickle.dump(speed_samples_add_triad, open(output_directory_address + 'speed_samples_' + 'add_triad_'
                                                      + prefix + ID + '.pkl', 'wb'))


        if settings.load_computations:
            speed_samples_add_random = pickle.load(open(output_directory_address + 'speed_samples_' + 'add_random_'
                                                        + prefix + ID + '.pkl', 'rb'))
            speed_samples_add_triad = pickle.load(open(output_directory_address + 'speed_samples_' + 'add_triad_'
                                                       + prefix + ID + '.pkl', 'rb'))

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
                            + prefix + ID + '.png')
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

            df[prefix + ID + '_add_random_' + str(percent_more_edges) + '_percent'] = pd.Series(speed_samples_add_random)
            df[prefix + ID + '_add_triad_' + str(percent_more_edges) + '_percent'] = pd.Series(speed_samples_add_triad)
            print(df)

    if settings.data_dump:
        df.to_csv(output_directory_address + prefix + 'data_dump.csv', index=False)#  , index=False


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
