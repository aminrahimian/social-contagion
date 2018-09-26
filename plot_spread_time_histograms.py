# Plotting histograms for spreading time samples that are computed for real networks

from models import *

size_of_dataset = 200

intervention_size_list = [5, 10, 15, 20, 25]


def check_type(obj):
    if isinstance(obj, np.generic):
        return np.asscalar(obj)
    else:
        return obj


if __name__ == '__main__':

    assert do_plots and load_computations, "We should be in do_plots and load_computations mode!"

    for intervention_size in intervention_size_list:
        for network_id in network_id_list:

            # load data:

            speed_samples_original = pickle.load(open(spreading_pickled_samples_directory_address
                                                      + 'speed_samples_original_'
                                                      + network_group + network_id
                                                      + model_id + '.pkl', 'rb'))
            speed_samples_add_random = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                        + str(intervention_size) + '_percent_' + 'add_random_'
                                                        + network_group + network_id
                                                        + model_id + '.pkl', 'rb'))
            speed_samples_add_triad = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                       + str(intervention_size) + '_percent_' + 'add_triad_'
                                                       + network_group + network_id
                                                       + model_id + '.pkl', 'rb'))

            speed_add_triad = np.mean(speed_samples_add_triad)

            speed_add_random = np.mean(speed_samples_add_random)

            speed_original = np.mean(speed_samples_original)

            std_add_triad = np.std(speed_samples_add_triad)

            std_add_random = np.std(speed_samples_add_random)

            std_original = np.std(speed_samples_original)

            speed_samples_rewired = pickle.load(open(spreading_pickled_samples_directory_address + 'speed_samples_'
                                                     + str(intervention_size) +
                                                     '_percent_rewiring_' + network_group + network_id
                                                     + model_id + '.pkl', 'rb'))

            speed_rewired = np.mean(speed_samples_rewired)

            std_rewired = np.std(speed_samples_rewired)

            # plot edge addition spreading times

            plt.figure()

            plt.hist([speed_samples_original, speed_samples_add_random, speed_samples_add_triad],
                     label=['original', 'random addition', 'triad addition'])

            plt.ylabel('Frequency')
            plt.xlabel('Time to Spread')
            plt.title('\centering The mean spread times are '
                      + str(Decimal(check_type(speed_original)).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(check_type(std_original)).quantize(TWOPLACES)) + ')'
                      + str(Decimal(check_type(speed_add_random)).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(check_type(std_add_random)).quantize(TWOPLACES)) + ')'
                      + ' and '
                      + str(Decimal(check_type(speed_add_triad)).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(check_type(std_add_triad)).quantize(TWOPLACES)) + '),'
                      + '\\vspace{-10pt}  \\begin{center}  in the original network and \\& with '
                      + str(intervention_size)
                      + '\% additional random or triad closing edges. \\end{center}')
            plt.legend()
            if show_plots:
                plt.show()

            if save_plots:
                plt.savefig(output_directory_address + 'speed_samples_histogram_'
                            + str(intervention_size) + '_edge_additions_'
                            + network_group + network_id
                            + model_id + '.png')

            plt.close()

            # plot rewiring spreading times

            plt.figure()

            plt.hist([speed_samples_original, speed_samples_rewired], label=['original', 'rewired'])

            plt.ylabel('Frequency')

            plt.xlabel('Time to Spread')

            plt.title('\centering The mean spread times are '
                      + str(Decimal(check_type(speed_original)).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(check_type(std_original)).quantize(TWOPLACES)) + ')'
                      + ' and '
                      + str(Decimal(check_type(speed_rewired)).quantize(TWOPLACES))
                      + '(SD=' + str(Decimal(check_type(std_rewired)).quantize(TWOPLACES)) + '),'
                      + '\\vspace{-10pt}  \\begin{center}  in the original and ' + str(intervention_size)
                      + '\% rewired network. \\end{center}')

            plt.legend()

            if show_plots:
                plt.show()

            if save_plots:
                plt.savefig(output_directory_address + 'speed_samples_histogram_' + str(intervention_size)
                            + '_percent_rewiring_' + network_group + network_id
                            + model_id + '.png')
            plt.close()
