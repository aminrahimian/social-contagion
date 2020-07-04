# plotting theory simulations for c_1 c_2 interpolation (Theorem 3).


from models import *

from computing_spread_time_c1_c2_interpolation import etas, all_q_labels, all_qs, size_of_dataset, \
    q_labels_old, q_labels  #, network_size

if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert (simulation_type == 'c1_c2_interpolation') \
           or (simulation_type =='c1_c2_interpolation_SimpleOnlyAlongC1'), \
        "we are not in the right simulation_type:" + simulation_type

    avg_spread_times = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_avg_'
                                        + simulation_type
                                        + '.pkl', 'rb'))
    std_spread_times = pickle.load(open(theory_simulation_pickle_address
                                        + 'spreading_time_std_'
                                        + simulation_type
                                        + '.pkl', 'rb'))

    plt.figure(1,(9.2, 8))

    for q_label in all_q_labels:
        print(q_label)
        if simulation_type == 'c1_c2_interpolation_SimpleOnlyAlongC1':
            if q_label not in q_labels_old+q_labels:
                print(q_label)
                print(etas)
                print(avg_spread_times[all_q_labels.index(q_label)])

                plt.errorbar(etas, avg_spread_times[all_q_labels.index(q_label)],
                             yerr=[1.96 * s / np.sqrt(size_of_dataset)
                                   for s in std_spread_times[all_q_labels.index(q_label)]],
                             linewidth=2.5, label='$q = '
                                                  + str(Decimal(all_qs[all_q_labels.index(q_label)]).quantize(FOURPLACES))
                                                  + '$')
        elif simulation_type == 'c1_c2_interpolation':
            if q_label in q_labels_old + q_labels:
                print(q_label)
                print(etas)
                print(avg_spread_times[all_q_labels.index(q_label)])

                plt.errorbar(etas, avg_spread_times[all_q_labels.index(q_label)],
                             yerr=[1.96 * s / np.sqrt(size_of_dataset)
                                   for s in std_spread_times[all_q_labels.index(q_label)]],
                             linewidth=2.5, label='$q = '
                                                  + str(
                        Decimal(all_qs[all_q_labels.index(q_label)]).quantize(FOURPLACES))
                                                  + '$')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('time to spread', fontsize=20)
    plt.xlabel('edges rewired ($\\eta$)', fontsize=20)
    # plt.title('\centering Complex Contagion over $\mathcal{C}_2^{\eta}, n = ' + str(network_size) + '$'
    #           '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions $(q)$ '
    #           'and Rewiring $(\eta)$   \\end{center}')
    if simulation_type == 'c1_c2_interpolation_SimpleOnlyAlongC1':
        plt.legend(loc='upper left', prop={'size': 20})#bbox_to_anchor=(1, 0.065),
    elif simulation_type == 'c1_c2_interpolation':
        plt.legend(loc='lower right', bbox_to_anchor=(1, 0.065), prop={'size': 18})  #
    if show_plots:
        plt.show()
    if save_plots:
        # plt.savefig(theory_simulation_output_address + simulation_type + '.png')
        if simulation_type is 'c1_c2_interpolation_SimpleOnlyAlongC1':
            plt.savefig('./figures/' + 'figure3A.pdf')
        elif simulation_type == 'c1_c2_interpolation':
            plt.savefig('./figures/' + 'figureS7.pdf')
