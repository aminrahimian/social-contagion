import networkx as NX
import math
import matplotlib.pyplot as plt
import random as RD
from models import cycle_union_Erdos_Renyi

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# def maslov_sneppen_rewiring(G):
#     r"""
#         Rewire the network graph according to the Maslov and
#         Sneppen method for degree-preserving random rewiring of a complex network,
#         as described on
#         `Maslov's webpage <http://www.cmth.bnl.gov/~maslov/matlab.htm>`_.
#         Return the resulting graph.
#         If a positive integer ``num_steps`` is given, then perform ``num_steps``
#         number of steps of the method.
#         Otherwise perform the default number of steps of the method, namely
#         ``4*graph.num_edges()`` steps.
#         The code is adopted from: https://github.com/araichev/graph_dynamics/blob/master/graph_dynamics.py
#         """
#
#     num_steps = 10 * G.number_of_edges()
#     for i in range(num_steps):
#         chosen_edges = RD.sample(G.edges(), 2)
#         e1 = chosen_edges[0]
#         e2 = chosen_edges[1]
#         new_e1 = (e1[0], e2[1])
#         new_e2 = (e2[0], e1[1])
#         if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1] or \
#                 G.has_edge(*new_e1) or G.has_edge(*new_e2):
#             # Not allowed to rewire e1 and e2. Skip.
#             continue
#         G.remove_edge(*e1)
#         G.remove_edge(*e2)
#         G.add_edge(*new_e1)
#         G.add_edge(*new_e2)


#
# params = {
#         'size': 10,  # populationSize,
#         'nearest_neighbors': 4,
#         'rewiring_probability': 0,
#     }
plt.figure(1)
G = cycle_union_Erdos_Renyi(10, k=2, c=2, seed=None)
# G = NX.connected_watts_strogatz_graph(params['size'], params['nearest_neighbors'], params['rewiring_probability'])
NX.draw_circular(G)
plt.text(-0.25,0,'$\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()


plt.figure(2)

G = cycle_union_Erdos_Renyi(10, k=4, c=0, seed=None)
# G = NX.connected_watts_strogatz_graph(params['size'], params['nearest_neighbors'], params['rewiring_probability'])
NX.draw_circular(G)
plt.text(-0.05,0,'$\mathcal{C}_{2}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()


plt.figure(3)

G = cycle_union_Erdos_Renyi(10, k=6, c=1, seed=None)
# G = NX.connected_watts_strogatz_graph(params['size'], params['nearest_neighbors'], params['rewiring_probability'])
NX.draw_circular(G)
plt.text(-0.25,0,'$\mathcal{C}_{3} \\cup \mathcal{G}_{n,1/n}$', fontsize=30,
         fontweight='bold', fontdict=None, withdash=False)

plt.show()

# maslov_sneppen_rewiring(G)
# NX.draw_circular(G)
# plt.show()

#
#
# params = {
#         'size': 10,  # populationSize,
#         'degree': 4,
#         'network_model': 'random_regular',
#     }
#
# G = NX.random_regular_graph(params['degree'],params['size'])
#
# NX.draw_circular(G)

# plt.show()