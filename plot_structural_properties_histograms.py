# plotting histograms for measured structural properties, treating each network id / intervention as a sample.

from models import *


intervention_size_list = [5, 10, 15, 20, 25]

old_properties = ['avg_clustering','average_shortest_path_length', 'diameter', 'size_2_core']

new_properties = ['avg_degree','diam_2_core', 'max_degree', 'min_degree',
                  'max_degree_2_core', 'min_degree_2_core',
                  'avg_degree_2_core', 'number_edges','number_edges_2_core',
                  'avg_clustering_2_core', 'transitivity', 'transitivity_2_core']

all_properties = old_properties + new_properties

included_properties = all_properties


if __name__ == '__main__':

    assert do_plots and load_computations, "we should be in do_plots and load_computations mode!"

    original_properties = [[] for ii in range(len(included_properties))]

    for network_id in network_id_list:

        print(network_id)

        #  load in the network and extract preliminary data

        fh = open(edgelist_directory_address + network_group + network_id + '.txt', 'rb')

        G = NX.read_edgelist(fh, delimiter=DELIMITER)

        print('original size ', len(G.nodes()))

        #  get the largest connected component:
        if not NX.is_connected(G):
            G = max(NX.connected_component_subgraphs(G), key=len)
            print('largest connected component extracted with size ', len(G.nodes()))

        #  remove self loops:
        if len(list(G.selfloop_edges())) > 0:
            print('warning the graph has ' + str(len(list(G.selfloop_edges()))) + ' self-loops that will be removed')
            print('number of edges before self loop removal: ', G.size())
            G.remove_edges_from(G.selfloop_edges())
            print('number of edges before self loop removal: ', G.size())

        network_size = NX.number_of_nodes(G)

        # number_edges = G.number_of_edges()

        G_list = [G]
        for included_property in included_properties:
            original_properties[included_properties.index(included_property)] += \
                measure_property(G_list, included_property)
            print(included_property, original_properties[included_properties.index(included_property)][-1])
        print(network_id_list.index(network_id))
        print(original_properties)
        print([this_property[network_id_list.index(network_id)] for this_property in original_properties])

    # interventions

    for intervention_size in intervention_size_list:

        print('intervention size:', intervention_size)

        mean_properties_add_random = [[] for i in range(len(included_properties))]
        mean_properties_add_triad = [[] for i in range(len(included_properties))]
        mean_properties_rewired = [[] for i in range(len(included_properties))]

        std_properties_add_random = [[] for i in range(len(included_properties))]
        std_properties_add_triad = [[] for i in range(len(included_properties))]
        std_properties_rewired = [[] for i in range(len(included_properties))]

        for network_id in network_id_list:

            print('network id:', network_id)

            properties_sample_add_random = []
            properties_sample_add_triad = []
            properties_sample_rewired = []

            for included_property in included_properties:

                properties_sample_add_random += \
                    [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                      + str(intervention_size) + '_percent_' + 'add_random_'
                                      + network_group + network_id + '.pkl', 'rb'))]

                properties_sample_add_triad += \
                    [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                      + str(intervention_size) + '_percent_' + 'add_triad_'
                                      + network_group + network_id + '.pkl', 'rb'))]

                properties_sample_rewired += \
                    [pickle.load(open(properties_pickled_samples_directory_address + included_property + '_'
                                      + str(intervention_size) + '_percent_' + 'rewiring_'
                                      + network_group + network_id + '.pkl', 'rb'))]

                mean_properties_add_random[included_properties.index(included_property)] \
                    += [np.mean(properties_sample_add_random[-1])]

                mean_properties_add_triad[included_properties.index(included_property)] \
                    += [np.mean(properties_sample_add_triad[-1])]

                mean_properties_rewired[included_properties.index(included_property)] \
                    += [np.mean(properties_sample_rewired[-1])]

                std_properties_add_random[included_properties.index(included_property)] \
                    += [np.std(properties_sample_add_random[-1])]

                std_properties_add_triad[included_properties.index(included_property)] \
                    += [np.std(properties_sample_add_triad[-1])]

                std_properties_rewired[included_properties.index(included_property)] \
                    += [np.std(properties_sample_rewired[-1])]

                print('loaded ' + included_property + '_'
                      + str(intervention_size) + '_percent_' + 'interventions_'
                      + network_group + network_id)

                print('loaded all properties.')

        for included_property in included_properties:

            plt.figure()

            iii = included_properties.index(included_property)

            plt.hist([original_properties[iii],
                      mean_properties_add_random[iii],
                      mean_properties_add_triad[iii],
                      mean_properties_rewired[iii]],
                     label=['original',
                            str(intervention_size) + ' \% random edge addition',
                            str(intervention_size) + ' \% triad edge addition',
                            str(intervention_size) + ' \% rewiring'])

            plt.ylabel('Frequency')
            plt.xlabel(included_property.replace("_", " "))
            plt.title(included_property.replace("_", " ") + ' under Various Interventions, Intervention Size: '
                      + str(intervention_size) + '%')
            plt.legend()
            if show_plots:
                plt.show()
            if save_plots:
                plt.savefig(output_directory_address + included_property + '_'
                            + 'intervention_size_' + str(intervention_size)+'.png')
            plt.close()
