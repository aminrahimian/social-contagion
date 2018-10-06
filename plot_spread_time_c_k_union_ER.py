# plotting the computed spread times in computing_spread_time_c_k_union_ER

from settings import *
from computing_spread_time_ck_union_ER import network_sizes, \
    number_of_cycle_neighbors_list, \
    size_of_dataset

if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert (simulation_type == 'ck_union_ER_vs_size') or \
           (simulation_type == 'ck_union_ER_vs_k'), \
        "we are not in the right simulation_type:" + simulation_type

    spread_time_avgs = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_avg_'
                                        + simulation_type
                                        + '.pkl', 'rb'))

    spread_time_stds = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_std_'
                                        + simulation_type
                                        + '.pkl', 'rb'))

    plt.figure()

    if simulation_type == 'ck_union_ER_vs_size':

        for number_of_cycle_neighbors in number_of_cycle_neighbors_list:
            number_of_cycle_neighbors_index = number_of_cycle_neighbors_list.index(number_of_cycle_neighbors)
            plt.errorbar(network_sizes,
                         (np.asarray(spread_time_avgs[number_of_cycle_neighbors_index]) /
                          (np.asarray(network_sizes) ** (3 / 4))),
                         yerr=(np.asarray([1.96*s/np.sqrt(size_of_dataset) for s in
                               spread_time_stds[number_of_cycle_neighbors_index]])/
                               (np.asarray(network_sizes) ** (3 / 4))),
                         linewidth=1.5,
                         label='$k = ' + str(number_of_cycle_neighbors // 2) + '$')

        plt.ylabel('time to spread (normalized by $\\times n^{-3/4}$)', fontsize=15)

        plt.xlabel('network size ($n$)', fontsize=15)
        # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (13 - 2k)/n$ \\end{center}', fontsize=18)
        plt.legend()

        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig(theory_simulation_output_address + simulation_type + '.png')

    elif simulation_type == 'ck_union_ER_vs_k':

        for network_size in network_sizes:
            network_size_index = network_sizes.index(network_size)
            plt.errorbar([x//2 for x in number_of_cycle_neighbors_list],
                         spread_time_avgs[network_size_index],
                         yerr=[1.96 * s / np.sqrt(size_of_dataset) for s in spread_time_stds[network_size_index]],
                         linewidth=1.5,
                         label='$n = ' + str(network_size) + '$')

        plt.ylabel('time to spread', fontsize=15)

        plt.xlabel('order of cycle-power ($k$)', fontsize=15)
        # plt.title('\centering Complex Contagions  over $\mathcal{C}_{k} \\cup \mathcal{G}_{n,p_n}$'
        #           '\\vspace{-10pt}  \\begin{center}  with $p_n = (15 - 2k)/n$ \\end{center}', fontsize=18)
        plt.legend()

        if show_plots:
            plt.show()
        if save_plots:
            plt.savefig(theory_simulation_output_address + simulation_type + '.png')
    else:

        print("simulation_type not set properly: " + simulation_type)
        exit()
