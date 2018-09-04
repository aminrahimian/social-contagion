# Measures the correct asymptotic rate of the spread for complex contagion in a specific variation of the Watts-Newman
# small world random graph model. We use the (0,1) threshold model for complex contagion with theta = 2, if two
# neighbors are infected then the agent gets infected. If less than two neighbors are infected then the agent does not
# get infected.

from models import *

import pylab as PL
import pycxsimulator

assert settings.simulator_mode, "we should be in simulator mode!"

RD.seed()


def init_viz():
    global positions, time, time_networks, labeldict
    time = 0
    if settings.layout == 'circular':
        positions = NX.circular_layout(time_networks[time], scale=4)
    elif settings.layout == 'spring':
        positions = NX.spring_layout(time_networks[time], scale=4)
    # set position to the network
    for t in range(len(time_networks)):
        for name, pos in positions.items():
            time_networks[t].node[name]['position'] = pos

def draw():
    global positions, time, time_networks

    PL.cla()
    NX.draw(time_networks[time],
            pos=positions,
            node_color=[time_networks[time].node[i]['state']+0.75 for i in time_networks[time].nodes()],
            with_labels=False,
            edge_color='c',
            cmap=PL.cm.YlOrRd,
            vmin=0,
            vmax=1)
    PL.axis('image')
    PL.title('t = ' + str(time))
    PL.savefig('./data/real_net/' + str(time) + '.png')

def step_viz():
    global time
    if time < len(time_networks) - 1:
        time += 1


if __name__ == '__main__':

    if settings.do_computations:
        dynamics = DeterministicLinear(settings.simulator_params)
        _, network_time_series = dynamics.time_the_total_spread(get_network_time_series=True, verbose = True)
        print(network_time_series)

    if settings.save_computations:
        pickle.dump(network_time_series, open(pickled_samples_directory_address
                                              + 'simulator_network_time_series_'
                                              + simulator_ID
                                              + '.pkl', 'wb'))

    if settings.load_computations:
        network_time_series = pickle.load(open(pickled_samples_directory_address
                                              + 'simulator_network_time_series_'
                                              + simulator_ID
                                              + '.pkl', 'rb'))

    # visualize time series
    global time_networks
    time_networks = network_time_series

    pycxsimulator.GUI().start(func=[init_viz, draw, step_viz])