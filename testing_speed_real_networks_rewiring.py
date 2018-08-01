# Comparing the rate of contagion in the original and rewired network.
# uses maslov_sneppen_rewiring(num_steps = np.floor(0.1 * self.params['network'].number_of_edges()))
# to rewire the network
# uses avg_speed_of_spread(dataset_size=1000,cap=0.9, mode='max') to measure the rate of spread
from settings import *
import settings

from models import *

size_of_dataset = 200

ID = 'cai_edgelist_4'

rewiring_percentage = 5

if __name__ == '__main__':

    if settings.do_computations:

        fh = open('./data/cai-data/edgelists/'+ID+'.txt', 'rb')

        G = NX.read_edgelist(fh)

        print(NX.is_connected(G))
        # print(type(G))
        #
        # add_edges(G, number_of_edges_to_be_added=10, mode='triadic_closures', seed=None)
        #
        # exit()

        network_size = NX.number_of_nodes(G)

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

        pickle.dump(speed_samples_rewired, open('./data/speed_samples_' + str(rewiring_percentage) +
                                                '_percent_rewiring_'+ID+'.pkl', 'wb'))
        pickle.dump(speed_samples_original, open('./data/speed_samples_original_' + ID + '.pkl', 'wb'))

    if settings.load_computations:

        speed_samples_rewired = pickle.load(open('./data/speed_samples_' + str(rewiring_percentage) +
                                                 '_percent_rewiring_'+ID+'.pkl', 'rb'))
        speed_samples_original = pickle.load(open('./data/speed_samples_original_' + ID + '.pkl', 'rb'))


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
            plt.savefig('./data/' + 'speed_samples_histogram_'+str(rewiring_percentage)+'_percent_rewiring_' + ID +  '.png')