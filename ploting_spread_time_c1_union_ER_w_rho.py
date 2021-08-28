# plotting theory simulations for c_1 union_ER versus delta (Proposition 2).

from models import *


from matplotlib import rc
rc("text", usetex=False)

from computing_spread_time_c1_union_ER_w_rho import rhos, qs, size_of_dataset, q_labels

if __name__ == '__main__':

    CUT = len([0.0, 0.0020408163265306124, 0.0040816326530612249, 0.0061224489795918373,
               0.0081632653061224497]) # 0.010204081632653062 erased number

    assert do_plots and load_computations, "we should be in load_computations and do_plots mode!"

    assert simulation_type == 'c1_union_ER_with_rho', \
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

    # overlay the plots for each q
    for q_label in q_labels:

        print(rhos[:CUT])
        print(avg_spread_times[q_labels.index(q_label)][:CUT])

        plt.errorbar(rhos[:CUT], avg_spread_times[q_labels.index(q_label)][:CUT],
                     yerr=[1.96 * s / np.sqrt(size_of_dataset)
                           for s in std_spread_times[q_labels.index(q_label)][:CUT]],
                     linewidth=2.5, label='$q = '
                                          + str(Decimal(qs[q_labels.index(q_label)]).quantize(FOURPLACES))
                                          + '$')

    plt.plot(rhos[:CUT],
             [10] * CUT,
             color='c', linewidth=1,linestyle='dashed')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.ylabel('time to spread', fontsize=20)
    plt.xlabel('Reevaluation probability ($\\rho$)', fontsize=20)
    # plt.title('\centering Complex Contagion over $\mathcal{C}_2^{\eta}, n = ' + str(network_size) + '$'
    #           '\\vspace{-10pt}  \\begin{center}  with Sub-threshold Adoptions $(q)$ '
    #           'and Rewiring $(\eta)$   \\end{center}')

    plt.legend(loc='upper left', fontsize=20)#bbox_to_anchor=(1, 0.065),
    # elif simulation_type == 'c1_c2_interpolation':
    #     plt.legend(loc='lower right', bbox_to_anchor=(1, 0.065), fontsize=11)  #
    if show_plots:
        plt.show()
    if save_plots:
        plt.savefig(theory_simulation_output_address + simulation_type + '.pdf')
        plt.savefig('./figures/' + 'figure3C.pdf')
