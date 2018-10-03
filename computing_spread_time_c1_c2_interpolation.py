# Computing the spread times in C_2^eta model. We use the (q,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If one neighbor is infected then the agent
# gets infected with probability q. eta determines how much of the C_2\C_1 edges are rewired.


from models import *


size_of_dataset = 1000

network_size = 500

etas = list(np.linspace(0, 100, 50))  # list(np.linspace(0, 80, 10))

qs_old = [1 / network_size ** x for x in [0.74,0.76,0.78]] # these are previously collected samples
#  their pkl files already exist

qs = [1 / network_size ** x for x in [0.6, 0.7]]# these are previously collected samples
#  their pkl files already exist

qs_new = [1 / network_size ** x for x in [0.4, 0.5]]


qs_new_new = [1 / network_size ** x for x in [0.25, 0.3]]

# [1 / network_size ** x for x in [0.1, 0.35, 0.6, 0.7, 0.8]]
#[1 / network_size ** x for x in [0.1,0.6,0.8]]#

q_labels_old = ['00'+str(x) for x in range(len(qs_old))]

q_labels = [str(x) for x in range(len(qs))]

q_labels_new = ['000'+str(x) for x in range(len(qs_new))]

q_labels_new_new = ['0000'+str(x) for x in range(len(qs_new))]

eta_labels = [str(x) for x in range(len(etas))]

all_q_labels = q_labels_old[::-1] + q_labels[::-1] + q_labels_new[::-1]+ q_labels_new_new[::-1]

# all_q_labels.reverse()

all_qs = qs_old[::-1] + qs[::-1] + qs_new[::-1] + qs_new_new[::-1]

# all_qs.reverse()

supply_the_exponentials = False

if supply_the_exponentials:
    add_long_ties_exp = np.random.exponential(scale=network_size ** 2,
                                              size=int(1.0 * network_size * (network_size - 1)) // 2)
    remove_cycle_edges_exp = np.random.exponential(scale=2 * network_size, size=network_size)


def compute_spread_time_for_q_eta(q, eta):
    if supply_the_exponentials:
        params = {
            'zero_at_zero': True,
            'network_model': 'c_1_c_2_interpolation',
            'size': network_size,
            'initial_states': [infected*active] + [infected*active] + [susceptible] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'fixed_prob_high': 1.0,
            'fixed_prob': q,
            'theta': 2,
            'eta': eta,
            'rewire': False,
            'add_long_ties_exp': add_long_ties_exp,
            'remove_cycle_edges_exp': remove_cycle_edges_exp,
        }
    else:
        params = {
            'zero_at_zero': True,
            'network_model': 'c_1_c_2_interpolation',
            'size': network_size,
            'initial_states': [infected * active] + [infected * active] + [susceptible] * (network_size - 2),
            # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'fixed_prob_high': 1.0,
            'fixed_prob': q,
            'theta': 2,
            'eta': eta,
            'rewire': False,
        }

    print('eta: ', params['eta'], 'q: ', params['fixed_prob'])

    if simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1':
        dynamics = SimpleOnlyAlongC1(params)
    elif simulation_type is 'c1_c2_interpolation':
        dynamics = DeterministicLinear(params)

    spread_time_avg, spread_time_std, _, _, samples = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    print('spread_time_avg: ', spread_time_avg)
    print('spread_time_std: ', spread_time_std)

    if save_computations:
        pickle.dump(spread_time_avg, open(theory_simulation_pickle_address
                                          + 'spreading_time_avg'
                                          + '_eta_' + eta_labels[etas.index(eta)] + '_q_'
                                          + q_labels_new[qs_new.index(q)]
                                          + '.pkl', 'wb'))
        pickle.dump(spread_time_std, open(theory_simulation_pickle_address
                                          + 'spreading_time_std'
                                          + '_eta_' + eta_labels[etas.index(eta)]
                                          + '_q_' + q_labels_new[qs_new.index(q)]
                                          + '.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations, "we should be in do_computations mode!"

    assert (simulation_type == 'c1_c2_interpolation_SimpleOnlyAlongC1') or \
           (simulation_type == 'c1_c2_interpolation'), \
        "simulation_type not set properly: " + simulation_type

    if do_multiprocessing:
        with multiprocessing.Pool(processes=number_CPU) as pool:
            pool.starmap(compute_spread_time_for_q_eta, product(qs_new_new, etas))
            pool.close()
            pool.join()
    else:  # no multi-processing:
        for q in qs_new_new:
            for eta in etas:
                compute_spread_time_for_q_eta(q, eta)

    if save_computations:
        avg_spread_times = []
        std_spread_times = []
        for q_label in all_q_labels:
            spread_avg = []
            spread_std = []
            for eta in etas:
                spread_time_avg = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_avg'
                                                   + '_eta_' + eta_labels[etas.index(eta)]
                                                   + '_q_' + q_label
                                                   + '.pkl', 'rb'))
                spread_time_std = pickle.load(open(theory_simulation_pickle_address
                                                   + 'spreading_time_std'
                                                   + '_eta_' + eta_labels[etas.index(eta)]
                                                   + '_q_' + q_label
                                                   + '.pkl', 'rb'))

                spread_avg.append(spread_time_avg)
                spread_std.append(spread_time_std)

            print(spread_avg)
            print(spread_std)

            avg_spread_times.append(spread_avg)

            std_spread_times.append(spread_std)

        print(avg_spread_times)
        print(std_spread_times)

        pickle.dump(avg_spread_times, open(theory_simulation_pickle_address
                                           + 'spreading_time_avg_'
                                           + simulation_type
                                           + '.pkl', 'wb'))
        pickle.dump(std_spread_times, open(theory_simulation_pickle_address
                                           + 'spreading_time_std_'
                                           + simulation_type
                                           + '.pkl', 'wb'))
