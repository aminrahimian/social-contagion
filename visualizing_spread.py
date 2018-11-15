# Network contagion simulator using pycxsimulator.GUI

from models import *


def init_viz():
    global positions, time, time_networks, labeldict
    time = 0
    if layout == 'circular':
        positions = NX.circular_layout(time_networks[time], scale=4)
    elif layout == 'spring':
        positions = NX.spring_layout(time_networks[time], scale=4)
    # set position to the network
    for t in range(len(time_networks)):
        for name, pos in positions.items():
            time_networks[t].node[name]['position'] = pos


def draw():
    global positions, time, time_networks, newly_infected_nodes, previously_infected_nodes

    PL.cla()
    NX.draw(time_networks[time],
            pos=positions,
            node_color=[time_networks[time].node[i]['state']*2  for i in time_networks[time].nodes()],
            with_labels=False,
            edge_color='c',
            cmap=PL.cm.YlOrRd,
            vmin=0,
            vmax=1)
    if highlight_infecting_edges:
        G = copy.deepcopy(time_networks[time])
        infected_nodes = [x for x, y in time_networks[time].nodes(data=True) if y['state'] == infected * active]
        if time == 0:
            newly_infected_nodes = []
            previously_infected_nodes = infected_nodes
        else:
            newly_infected_nodes = list(set(infected_nodes) - set(previously_infected_nodes))
            previously_infected_nodes = infected_nodes
        # edges_incident_to_infected_nodes = G.edges(infected_nodes)
        infecting_edges = [e for e in time_networks[time].edges if ((e[0] in newly_infected_nodes)
                                                                    and ((e[1] in previously_infected_nodes)) or
                                                                    (e[0] in previously_infected_nodes)
                                                                    and ((e[1] in newly_infected_nodes)))]
        # edges_between_infected_nodes = G.subgraph(infected_nodes).edges()
        # print(infected_nodes)
        # print(edges_between_infected_nodes)
        # print(infecting_edges)
        # NX.draw_networkx_nodes(G, positions, nodelist=infected_nodes, node_color='r')
        NX.draw_networkx_edges(G, positions, edgelist=infecting_edges, edge_color='r', width=1)

    PL.axis('image')

    if show_times:
        PL.title('t = ' + str(time))

    if save_snapshots:
        PL.savefig(visualizing_spread_output_address + str(time) + '.png', bbox_inches='tight')


def step_viz():
    global time
    if time < len(time_networks) - 1:
        time += 1


if __name__ == '__main__':

    assert simulator_mode, "we should be in simulator mode!"

    if do_computations:
        dynamics = DeterministicLinear(simulator_params)
        _, _, network_time_series, _ = dynamics.time_the_total_spread(get_time_series=True, verbose=True)
        print(network_time_series)

    if save_computations:
        pickle.dump(network_time_series, open(visualizing_spread_pickle_address
                                              + 'simulator_network_time_series_'
                                              + simulator_ID
                                              + '.pkl', 'wb'))

    if load_computations:
        network_time_series = pickle.load(open(visualizing_spread_pickle_address
                                               + 'simulator_network_time_series_'
                                               + simulator_ID
                                               + '.pkl', 'rb'))

        # convert the infected=1 state from old versions to infected*active = 0.5 states

        for i in range(len(network_time_series)):
            for node_i in network_time_series[i].nodes():
                if network_time_series[i].node[node_i]['state'] == 1:
                    network_time_series[i].node[node_i]['state'] = infected*active

    # visualize time series

    global time_networks
    time_networks = network_time_series

    pycxsimulator.GUI().start(func=[init_viz, draw, step_viz])