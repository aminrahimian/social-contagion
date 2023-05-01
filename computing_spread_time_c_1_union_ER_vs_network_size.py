# Spread time in C_1 uninon G(n,p_n) allowing for simple adoptions with probability (q) - theorem 2 simulations.
from models import *

size_of_dataset = 500

model = 'Threshold'

fixed_prob_low = [0.01, 0.02, 0.03]
NETWORK_SIZE = [100, 400, 900, 1600, 2500]


def compute_spread_time(NET_size, Q):
    if model == 'Threshold':
        params_mix = {
            'zero_at_zero': True,
            'network_model': 'two_d_lattice_union_Erdos_Renyi',
            'size': NET_size,  # populationSize,
            'initialization_mode': 'fixed_number_initial_infection_at_center',
            'initial_states': [infected * active] + [infected * active] + [susceptible] * (NET_size - 2),
            # two initial seeds, next to each other
            'delta': 0.0,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 2,
            'fixed_prob_high': 1.0,
            'fixed_prob': Q,
            'theta': 2,
            'c': 2,
            'rewire': False,
        }

        dynamics = DeterministicLinear(params_mix)

    else:
        assert False, "model is not supported for C_1 union random graph simulations"

    spread_time_avg, spread_time_std, _, _, _, _, _, _, _, _, _ = \
        dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

    adoption_probabilities = dynamics.get_activation_probabilities()

    q = adoption_probabilities[1]

    if save_computations:
        address = theory_simulation_pickle_address + str(NET_size) + '/'
        pickle.dump(spread_time_std, open(address
                                          + model + '_Q_'
                                          + str(fixed_prob_low.index(Q))
                                          + '_spread_time_std.pkl', 'wb'))
        pickle.dump(spread_time_avg, open(address
                                          + model + '_Q_'
                                          + str(fixed_prob_low.index(Q))
                                          + '_spread_time_avg.pkl', 'wb'))
        pickle.dump(q, open(address
                            + model + '_Q_'
                            + str(fixed_prob_low.index(Q))
                            + '_q.pkl', 'wb'))


if __name__ == '__main__':

    assert do_computations or data_dump, "we should be in do_computations or data_dump mode!"

    input_list = [(net_size, Q) for net_size in NETWORK_SIZE for Q in fixed_prob_low]

    if not data_dump:
        if do_multiprocessing:
            with multiprocessing.Pool(processes=number_CPU) as pool:
                pool.starmap(compute_spread_time, input_list)
            pool.close()
            pool.join()
        else:
            for net_size in NETWORK_SIZE:
                for Q in fixed_prob_low:
                    compute_spread_time(model, Q, net_size)
    elif data_dump:
        # load individual avg and std pkl files, organize them in a list and save them in a pair of pkl files
        # one for all the avg's and the other for all the std's.

        df = pd.DataFrame(columns=['network_size', 'time_to_spread', 'std_in_spread', 'q'])

        for size in NETWORK_SIZE:
            address = theory_simulation_pickle_address + str(size) + '/'
            qs = []
            spread_time_avgs = []
            spread_time_stds = []
            for Q in fixed_prob_low:
                index = fixed_prob_low.index(Q)

                spread_time_avg = pickle.load(open(address
                                                   + model + '_Q_'
                                                   + str(fixed_prob_low.index(Q))
                                                   + '_spread_time_avg.pkl', 'rb'))
                print(spread_time_avg)
                spread_time_std = pickle.load(open(address
                                                   + model + '_Q_'
                                                   + str(fixed_prob_low.index(Q))
                                                   + '_spread_time_std.pkl', 'rb'))
                print(spread_time_std)
                q = pickle.load(open(address
                                     + model + '_Q_'
                                     + str(fixed_prob_low.index(Q))
                                     + '_q.pkl', 'rb'))

                qs.append(q)

                spread_time_avgs.append(spread_time_avg)

                spread_time_stds.append(spread_time_avg)

                temp = {'network_size': size,
                        'time_to_spread': spread_time_avgs[index],
                        'std_in_spread': 1.96 * spread_time_stds[index] / np.sqrt(500),
                        'q': qs[index]}
                df = df.append(temp, ignore_index=True)
        pd.DataFrame(df).to_csv(theory_simulation_output_address
                                + simulation_type + 'spreading_data_dump.csv',
                                index=False)
