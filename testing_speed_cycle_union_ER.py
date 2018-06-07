# Comparing the speed in C_k uninon G(n,p_n). We use the (0,1) threshold model for complex contagion with theta = 2,
# if two neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent
# does not get infected.

from models import *
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

RD.seed()

size_of_dataset = 20


if __name__ == '__main__':

    network_sizes = [2000, 4000, 6000, 8000,10000]#[10]#list(np.int_(np.ceil((np.logspace(2.0, 5.0, num=10)))))
#[1000, 2000, 3000, 4000]#,5000,6000,7000,8000,9000,10000]#, 2000, 3000, 4000]
    speeds_12_1 = []
    speeds_std_12_1 = []
    speeds_10_3 = []
    speeds_std_10_3 = []
    speeds_8_5 = []
    speeds_std_8_5 = []

    for network_size in network_sizes:
        params_12_1 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size-2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 12,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 1,
        }
        print(params_12_1['size'])
        dynamics = DeterministicLinear(params_12_1)
        speed,std,_,_ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_12_1.append(speed)
        speeds_std_12_1.append(std)
        print(speeds_12_1)
        print(speeds_std_12_1)
        # print(np.asarray(speeds_12_1)/(np.asarray(network_sizes) ** (3 / 4)))

        params_10_3 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 10,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 3,
        }
        print(params_10_3['size'])
        dynamics = DeterministicLinear(params_10_3)
        speed,std,_,_ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_10_3.append(speed)
        speeds_std_10_3.append(std)
        print(speeds_10_3)
        print(speeds_std_10_3)
        # print(np.asarray(speeds_10_3) / (np.asarray(network_sizes) ** (3 / 4)))

        params_8_5 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': network_size,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': 8,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 5,
        }
        print(params_8_5['size'])
        dynamics = DeterministicLinear(params_8_5)
        speed,std,_,_ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_8_5.append(speed)
        speeds_std_8_5.append(std)
        print(speeds_8_5)
        print(speeds_std_8_5)
        # print(np.asarray(speeds_8_5) /(np.asarray(network_sizes) ** (3 / 4)))

    print(np.asarray(speeds_12_1))

    print(np.asarray(speeds_12_1) / (np.asarray(network_sizes) ** (3 / 4)))

    print((np.asarray(network_sizes) ** (3 / 4)))

    print(np.asarray(speeds_10_3))

    print(np.asarray(speeds_10_3) / (np.asarray(network_sizes) ** (3 / 4)))

    print(np.asarray(speeds_8_5) / (np.asarray(network_sizes) ** (3 / 4)))

    plt.figure(1)

    plt.errorbar(network_sizes, np.asarray(speeds_12_1)/np.asarray(network_sizes)**(3/4),
                 yerr=np.asarray(speeds_std_12_1)/(np.asarray(network_sizes)**(3/4)),
                 color='r', linewidth=2.5,
                 label='$k = {6}$')  # label='$C_{6} \\cup G_{n,1/n}$'
    plt.errorbar(network_sizes, np.asarray(speeds_10_3)/np.asarray(network_sizes)**(3/4),
                 yerr=np.asarray(speeds_std_10_3)/(np.asarray(network_sizes)**(3/4)),
                 color='g', linewidth=2.5,
                 label='$k = {5}$')  # label='$C_{5} \\cup G_{n,3/n}$'
    plt.errorbar(network_sizes, np.asarray(speeds_8_5)/np.asarray(network_sizes)**(3/4),
                 yerr=np.asarray(speeds_std_8_5)/(np.asarray(network_sizes)**(3/4)),
                 color='c', linewidth=2.5,
                 label='$k = {4}$')  # label='$C_{4}  \\cup G_{n,5/n}$'

    plt.ylabel('Normalized ($\\times n^{-3/4}$) Time to Spread')
    plt.xlabel('Network Size $(n)$')
    plt.title('Complex Contagions  over $C_{k} \\cup G_{n,p_n}$ with $p_n = (13 - 2k)/n$')
    plt.legend()
    # plt.show()
    plt.savefig('./data/' + 'speed_of_contagion_union_Net_Size.png')



    k = [14,12,10,8,6,4]

    speeds_1000 = []
    speeds_std_1000 = []

    for number_of_cycle_edges in k:
        params_1000 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': 1000,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': number_of_cycle_edges,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 15-number_of_cycle_edges,
        }
        print(params_1000['size'])
        dynamics = DeterministicLinear(params_1000)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_1000.append(speed)
        speeds_std_1000.append(std)
        print(speeds_1000)
        print(speeds_std_1000)

    speeds_3000 = []
    speeds_std_3000 = []

    for number_of_cycle_edges in k:
        params_3000 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': 3000,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': number_of_cycle_edges,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 15 - number_of_cycle_edges,
        }
        print(params_3000['size'])
        dynamics = DeterministicLinear(params_3000)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_3000.append(speed)
        speeds_std_3000.append(std)
        print(speeds_3000)
        print(speeds_std_3000)

    speeds_5000 = []
    speeds_std_5000 = []

    for number_of_cycle_edges in k:
        params_5000 = {
            'network_model': 'cycle_union_Erdos_Renyi',
            'size': 5000,  # populationSize,
            'initial_states': [1] + [1] + [0] * (network_size - 2),  # two initial seeds, next to each other
            'delta': 0.0000000000000001,  # recoveryProb,  # np.random.beta(5, 2, None), # recovery probability
            'nearest_neighbors': number_of_cycle_edges,
            'fixed_prob_high': 1.0,
            'fixed_prob': 0.0,
            'theta': 2,
            'c': 15 - number_of_cycle_edges,
        }
        print(params_5000['size'])
        dynamics = DeterministicLinear(params_5000)
        speed, std, _, _ = dynamics.avg_speed_of_spread(dataset_size=size_of_dataset, mode='total')
        speeds_5000.append(speed)
        speeds_std_5000.append(std)
        print(speeds_5000)
        print(speeds_std_5000)

    plt.figure(2)

    plt.errorbar(np.asarray(k)/2, speeds_1000, yerr=speeds_std_1000, color='r', linewidth=2.5,
                 label='$n = 1000$')

    plt.errorbar(np.asarray(k) / 2, speeds_3000, yerr=speeds_std_3000, color='g', linewidth=2.5,
                 label='$n = 3000$')

    plt.errorbar(np.asarray(k) / 2, speeds_5000, yerr=speeds_std_5000, color='c', linewidth=2.5,
                 label='$n = 5000$')

    plt.ylabel('Time to Spread')
    plt.xlabel('Number of Cycle Edges (k)')
    plt.title('Complex Contagions  over $C_{k} \\cup G_{n,(15-2k)/n}$ with $p_n = (15 - 2k)/n$')
    plt.legend()
    # plt.show()
    plt.savefig('./data/' + 'speed_of_contagion_union_Cycle_Edges.png')