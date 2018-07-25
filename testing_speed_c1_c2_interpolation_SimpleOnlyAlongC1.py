# Comparing the speed in C_k uninon G(n,p_n). We use the (0,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent
# does not get infected.

from settings import *
from models import *
import settings


size_of_dataset = 20

if __name__ == '__main__':

    network_size = 200

    speeds_alpha_10 = []
    speeds_alpha_60 = []
    speeds_alpha_80 = []
    # speeds_alpha_150= []

    speeds_std_alpha_10 = []
    speeds_std_alpha_60 = []
    speeds_std_alpha_80 = []
    # speeds_std_alpha_150 = []

    etas = np.linspace(0,80,10)#-1000000, 0.2,0.25,0.3,0.35,0.4,
    print(etas)

    add_long_ties_exp = np.random.exponential(scale=network_size ** 2,
                                              size=int(1.0 * network_size * (network_size - 1)) // 2)
    remove_cycle_edges_exp = np.random.exponential(scale=2 * network_size,
                                                   size=network_size)

    alpha_1 = 0.1
    alpha_2 = 0.6
    alpha_3 = 0.8
    # alpha_4 = 1.5

    if settings.do_computations:
        for eta in etas:

            params_alpha_10 = {
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

            print(params_alpha_10['eta'])

            dynamics = SimpleOnlyAlongC1(params_alpha_10)

            speed,std,_,_ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')

            speeds_alpha_10.append(speed)
            speeds_std_alpha_10.append(std)
            print(speeds_alpha_10)
            print(speeds_std_alpha_10)

            params_alpha_60 = {
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

            dynamics = SimpleOnlyAlongC1(params_alpha_60)
            speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            speeds_alpha_60.append(speed)
            speeds_std_alpha_60.append(std)
            print(speeds_alpha_60)
            print(speeds_std_alpha_60)

            params_alpha_80 = {
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

            dynamics = SimpleOnlyAlongC1(params_alpha_80)
            speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            speeds_alpha_80.append(speed)
            speeds_std_alpha_80.append(std)
            print(speeds_alpha_80)
            print(speeds_std_alpha_80)

            # params_alpha_150 = {
            #     'zero_at_zero': True,
            #     'network_model': 'c_1_c_2_interpolation',
            #     'size': network_size,  # populationSize,
            #     'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            #     'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            #     'fixed_prob_high': 1.0,
            #     'fixed_prob': 1 / network_size ** alpha_4,
            #     'theta': 2,
            #     'eta': eta,
            #     'add_long_ties_exp': add_long_ties_exp,
            #     'remove_cycle_edges_exp': remove_cycle_edges_exp,
            # }
            #
            # dynamics = SimpleOnlyAlongC1(params_alpha_150)
            # speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
            # speeds_alpha_150.append(speed)
            # speeds_std_alpha_150.append(std)
            # print(speeds_alpha_150)
            # print(speeds_std_alpha_150)

        if settings.save_computations:
            pickle.dump(speeds_alpha_10,open( './data/speeds_alpha_10.pkl','wb'))
            pickle.dump(speeds_alpha_60, open('./data/speeds_alpha_60.pkl', 'wb'))
            pickle.dump(speeds_alpha_80, open('./data/speeds_alpha_80.pkl', 'wb'))
            # pickle.dump(speeds_alpha_150, open('./data/speeds_alpha_150.pkl', 'wb'))

            pickle.dump(speeds_std_alpha_10,open( './data/speeds_std_alpha_10.pkl','wb'))
            pickle.dump(speeds_std_alpha_60, open('./data/speeds_std_alpha_60.pkl', 'wb'))
            pickle.dump(speeds_std_alpha_80, open('./data/speeds_std_alpha_80.pkl', 'wb'))
            # pickle.dump(speeds_std_alpha_150, open('./data/speeds_std_alpha_150.pkl', 'wb'))

    if settings.do_plots:

        if settings.load_computations:
            speeds_alpha_10 = pickle.load(open('./data/speeds_alpha_10.pkl','rb'))
            speeds_alpha_60 = pickle.load(open('./data/speeds_alpha_60.pkl', 'rb'))
            speeds_alpha_80 = pickle.load(open('./data/speeds_alpha_80.pkl', 'rb'))
            # speeds_alpha_150 = pickle.load(open('./data/speeds_alpha_150.pkl', 'rb'))

            speeds_std_alpha_10 = pickle.load(open( './data/speeds_std_alpha_10.pkl', 'rb'))
            speeds_std_alpha_60 = pickle.load(open('./data/speeds_std_alpha_60.pkl', 'rb'))
            speeds_std_alpha_80 = pickle.load(open('./data/speeds_std_alpha_80.pkl', 'rb'))
            # speeds_std_alpha_150 = pickle.load(open('./data/speeds_std_alpha_150.pkl', 'rb'))

        plt.figure(1)

        plt.errorbar(etas, speeds_alpha_10, yerr=speeds_std_alpha_10, color='r', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_1).quantize(FOURPLACES))+'$')

        plt.errorbar(etas, speeds_alpha_60, yerr=speeds_std_alpha_60, color='g', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_2).quantize(FOURPLACES))+'$')

        plt.errorbar(etas, speeds_alpha_80, yerr=speeds_std_alpha_80, color='c', linewidth=2.5,
                     label='$q_n = '+str(Decimal(1 / network_size ** alpha_3).quantize(FOURPLACES))+'$')

        # plt.errorbar(etas, speeds_alpha_150, yerr=speeds_std_alpha_150, color='c', linewidth=2.5,
        #              label='$q_n = ' + str(Decimal(1 / network_size ** alpha_4).quantize(FOURPLACES)) + '$')

        plt.ylabel('Time to Spread')
        plt.xlabel('Edges Rewired ($\\eta$)')
        plt.title('\centering Complex Contagion over $\mathcal{C}_1 \\cup \mathcal{G}_{\eta} '
                  '\\cup \mathcal{D}_{\eta},n = 200$'
                  '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions $(q_n)$ '
                  'and Rewiring $(\eta)$   \\end{center}')
        plt.legend()
        plt.show()
        if settings.save_plots:
            plt.savefig('./data/' + 'speed_of_contagion_c_1_c_2_interpolation.png')