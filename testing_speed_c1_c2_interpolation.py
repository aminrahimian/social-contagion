# Comparing the speed in C_k uninon G(n,p_n). We use the (0,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent
# does not get infected.

from settings import *
from models import *
import settings

size_of_dataset = 20

if __name__ == '__main__':

    # speeds_alpha_20 = pickle.load(open('./data/speeds_alpha_20.pkl', 'rb'))
    # print(speeds_alpha_20)

    network_size = 100

    speeds_alpha_20 = []
    # speeds_alpha_50 = []
    speeds_alpha_70 = []
    speeds_alpha_90 = []

    speeds_std_alpha_20 = []
    # speeds_std_alpha_50 = []
    # speeds_std_alpha_60 = []
    speeds_std_alpha_70 = []
    speeds_std_alpha_90 = []

    # etas = np.linspace(0,70,7)#-1000000, 0.2,0.25,0.3,0.35,0.4,
    etas = np.linspace(0, 250, 10)
    print(etas)

    add_long_ties_exp = np.random.exponential(scale=network_size ** 2,
                                              size=int(1.0 * network_size * (network_size - 1)) // 2)
    remove_cycle_edges_exp = np.random.exponential(scale=2 * network_size,
                                                   size=network_size)
    #
    # alpha_1 = 0.2
    # alpha_2 = 0.7
    # alpha_3 = 0.9


    alpha_1 = 0.05
    alpha_2 = 0.5
    alpha_3 = 0.7

    if settings.do_computations:
        for eta in etas:

            params_alpha_20 = {
                'zero_at_zero': True,
                'network_model': 'c_1_c_2_interpolation',
                'size': network_size,  # populationSize,
                'initial_states': [1] + [1] + [0] * (network_size-2),  # two initial seeds, next to each other
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': 1.0,
                'fixed_prob': 1/network_size **alpha_1,
                'theta': 2,
                'eta': eta,
                'add_long_ties_exp': add_long_ties_exp,
                'remove_cycle_edges_exp': remove_cycle_edges_exp,
            }

            print(params_alpha_20['eta'])

            dynamics = DeterministicLinear(params_alpha_20)

            speed,std,_,_ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            print(dynamics.params['network'].nodes())
            print(dynamics.params['network'].edges())
            speeds_alpha_20.append(speed)
            speeds_std_alpha_20.append(std)
            print(speeds_alpha_20)
            print(speeds_std_alpha_20)

            # alpha = 0.5
            #
            # params_alpha_50 = {
            #     'zero_at_zero': True,
            #     'network_model': 'c_1_c_2_interpolation',
            #     'size': network_size,  # populationSize,
            #     'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            #     'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            #     'fixed_prob_high': 1.0,
            #     'fixed_prob': 1 / network_size ** alpha,
            #     'theta': 2,
            #     'eta': network_size ** beta,
            #     'add_long_ties_exp': add_long_ties_exp,
            #     'remove_cycle_edges_exp': remove_cycle_edges_exp,
            # }
            #
            # dynamics = DeterministicLinear(params_alpha_50)
            # speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            # speeds_alpha_50.append(speed)
            # speeds_std_alpha_50.append(std)
            # print(speeds_alpha_50)
            # print(speeds_std_alpha_50)

            # alpha = 0.6
            #
            # params_alpha_60 = {
            #     'zero_at_zero': True,
            #     'network_model': 'c_1_c_2_interpolation',
            #     'size': network_size,  # populationSize,
            #     'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            #     'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            #     'fixed_prob_high': 1.0,
            #     'fixed_prob': 1 / network_size ** alpha,
            #     'theta': 2,
            #     'eta': network_size ** beta,
            #     'add_long_ties_exp': add_long_ties_exp,
            #     'remove_cycle_edges_exp': remove_cycle_edges_exp,
            # }
            #
            # dynamics = DeterministicLinear(params_alpha_60)
            # speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            # speeds_alpha_60.append(speed)
            # speeds_std_alpha_60.append(std)
            # print(speeds_alpha_60)
            # print(speeds_std_alpha_60)

            params_alpha_70 = {
                'zero_at_zero': True,
                'network_model': 'c_1_c_2_interpolation',
                'size': network_size,  # populationSize,
                'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': 1.0,
                'fixed_prob': 1 / network_size ** alpha_2,
                'theta': 2,
                'eta': eta,
                'add_long_ties_exp': add_long_ties_exp,
                'remove_cycle_edges_exp': remove_cycle_edges_exp,
            }

            dynamics = DeterministicLinear(params_alpha_70)
            speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            speeds_alpha_70.append(speed)
            speeds_std_alpha_70.append(std)
            print(speeds_alpha_70)
            print(speeds_std_alpha_70)

            params_alpha_90 = {
                'zero_at_zero': True,
                'network_model': 'c_1_c_2_interpolation',
                'size': network_size,  # populationSize,
                'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
                'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
                'fixed_prob_high': 1.0,
                'fixed_prob': 1 / network_size ** alpha_3,
                'theta': 2,
                'eta': eta,
                'add_long_ties_exp': add_long_ties_exp,
                'remove_cycle_edges_exp': remove_cycle_edges_exp,
            }

            dynamics = DeterministicLinear(params_alpha_90)
            speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            speeds_alpha_90.append(speed)
            speeds_std_alpha_90.append(std)
            print(speeds_alpha_90)
            print(speeds_std_alpha_90)


        if settings.save_computations:
            pickle.dump(speeds_alpha_20,open( './data/speeds_alpha_20.pkl','wb'))
            pickle.dump(speeds_alpha_70, open('./data/speeds_alpha_70.pkl', 'wb'))
            pickle.dump(speeds_alpha_90, open('./data/speeds_alpha_90.pkl', 'wb'))

            pickle.dump(speeds_std_alpha_20,open( './data/speeds_std_alpha_20.pkl','wb'))
            pickle.dump(speeds_std_alpha_70, open('./data/speeds_std_alpha_70.pkl', 'wb'))
            pickle.dump(speeds_std_alpha_90, open('./data/speeds_std_alpha_90.pkl', 'wb'))

    if settings.do_plots:

        if settings.load_computations:
            speeds_alpha_20 = pickle.load(open('./data/speeds_alpha_20.pkl','rb'))
            speeds_alpha_70 = pickle.load(open('./data/speeds_alpha_70.pkl', 'rb'))
            speeds_alpha_90 = pickle.load(open('./data/speeds_alpha_90.pkl', 'rb'))

            speeds_std_alpha_20 = pickle.load(open( './data/speeds_std_alpha_20.pkl', 'rb'))
            speeds_std_alpha_70 = pickle.load(open('./data/speeds_std_alpha_70.pkl', 'rb'))
            speeds_std_alpha_90 = pickle.load(open('./data/speeds_std_alpha_90.pkl', 'rb'))

        plt.figure(1)

        plt.errorbar(etas, speeds_alpha_20, yerr=speeds_std_alpha_20, color='r', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_1).quantize(FOURPLACES))+'$')
        #
        # plt.errorbar(network_size**np.asarray(betas), speeds_alpha_50,
        # yerr=speeds_std_alpha_50, color='g', linewidth=2.5,
        #              label='$\\alpha = 0.5$')

        # plt.errorbar(network_size ** np.asarray(betas), speeds_alpha_60,
        # yerr=speeds_std_alpha_60, color='c', linewidth=2.5,
        #              label='$\\alpha = 0.6$')

        plt.errorbar(etas, speeds_alpha_70, yerr=speeds_std_alpha_70, color='g', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_2).quantize(FOURPLACES))+'$')

        plt.errorbar(etas, speeds_alpha_90, yerr=speeds_std_alpha_90, color='b', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_3).quantize(FOURPLACES))+'$')

        plt.ylabel('Time to Spread')
        plt.xlabel('Edges Rewired ($\\eta$)')
        plt.title('\centering Complex Contagion over $\mathcal{C}_1 \\cup \mathcal{G}_{\eta} '
                  '\\cup \mathcal{D}_{\eta},n = 1000$'
                  '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions $(q_n)$ '
                  'and Rewiring $(\eta)$   \\end{center}')
        plt.legend()
        plt.show()
        if settings.save_plots:
            plt.savefig('./data/' + 'speed_of_contagion_c_1_c_2_interpolation.png')