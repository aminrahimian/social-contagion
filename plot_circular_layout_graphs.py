import networkx as NX
from models import cycle_union_Erdos_Renyi

import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
#
# plt.figure(1,(6,6))
# G = cycle_union_Erdos_Renyi(10, k=2, c=1, seed=None, color_the_edges=True, weight_the_edges=True)
# colors = [G[u][v]['color'] for u,v in G.edges]
# weights = [G[u][v]['weight'] for u,v in G.edges]
# NX.draw_circular(G, edge_color=colors, width=weights)
# plt.text(-0.25,0,'$\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n}$', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)

plt.figure(1,(6,6))

G = cycle_union_Erdos_Renyi(10, k=6, c=0, seed=None, color_the_edges=True, weight_the_edges=True)
G.add_edge(1,5,color='b')
G[1][5].update(weight=4)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
# plt.text(-0.25,0,'$\mathcal{C}_{3} \\cup \mathcal{G}_{n,1/n}$', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)
plt.text(-0.25,-0.1,'$\mathcal{C}_{3,1/n}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figureS1p1.pdf')

plt.figure(2,(6,6))

G = cycle_union_Erdos_Renyi(10, k=4, c=0, seed=None, color_the_edges=True, weight_the_edges=True)
# G = NX.connected_watts_strogatz_graph(params['size'], params['nearest_neighbors'], params['rewiring_probability'])
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
plt.text(-0.05,-0.1,'$\mathcal{C}_{2}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figureS1p2.pdf')

plt.figure(3,(6,6))

G = cycle_union_Erdos_Renyi(10, k=4, c=0, seed=None, color_the_edges=True, weight_the_edges=True)
G.add_edge(1,5,color='b')
G[1][5].update(weight=4)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
# plt.text(-0.25,0,'$\mathcal{C}_{2} \\cup \mathcal{G}_{n,1/n}$', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)
plt.text(-0.25,-0.1,'$\mathcal{C}_{2,1/n}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figureS1p3.pdf')

# plt.figure(4,(6,6))
# G = cycle_union_Erdos_Renyi(10, k=4, c=0, seed=None, color_the_edges=True, weight_the_edges=True)
# G.remove_edge(4,6)
# G.remove_edge(5,7)
# G.add_edge(1,5,color='b')
# G[1][5].update(weight=4)
# colors = [G[u][v]['color'] for u,v in G.edges]
# weights = [G[u][v]['weight'] for u,v in G.edges]
# NX.draw_circular(G, edge_color=colors, width=weights)
# # plt.text(-0,0,'$\mathcal{C}_{2}^{\eta} = $  '
# #               '\\\ $\mathcal{C}_1 \cup \mathcal{G}_{\eta} \cup \mathcal{D}_{\eta}$', fontsize=30,
# #          fontweight='bold', fontdict=None, withdash=False)
# plt.text(-0.1,-0.1,'$\mathcal{C}_{2}^{\eta}$', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)
# plt.savefig('./figures/figure2Bp1.pdf')


# C_1 C_2 interpolation figures
# C_2:



plt.figure(4,(6,6))
G = cycle_union_Erdos_Renyi(10, k=2, c=0, seed=None,
                            color_the_edges=True,
                            cycle_edge_color='k',
                            weight_the_edges=True,
                            cycle_edge_weights=8)
for i in range(10):
    head = i
    tail = (i+2)%10
    G.add_edge(head,tail,color='g')
    G[head][tail].update(weight=4)
G.add_edge(3,7,color='b', weight=4)
G.add_edge(1,8,color='b', weight=4)
G.add_edge(1,5,color='b', weight=4)
G.add_edge(7,0,color='b', weight=4)
G.add_edge(2,5,color='b', weight=4)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
plt.text(-0.25,-0.1,'$\mathcal{C}_{2,2/n}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figure2Bp1.pdf')


plt.figure(5,(6,6))
G = cycle_union_Erdos_Renyi(10, k=2, c=0, seed=None,
                            color_the_edges=True,
                            cycle_edge_color='k',
                            weight_the_edges=True,
                            cycle_edge_weights=8)
for i in range(10):
    head = i
    tail = (i+2)%10
    G.add_edge(head,tail,color='g')
    G[head][tail].update(weight=4)

colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)

plt.text(-0.1,-0.1,'$\mathcal{C}_{2}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figure2Bp2.pdf')
# C_2^eta

plt.figure(6,(6,6))
G = cycle_union_Erdos_Renyi(10, k=2, c=0, seed=None,
                            color_the_edges=True,
                            cycle_edge_color='k',
                            weight_the_edges=True,
                            cycle_edge_weights=8)
for i in range(10):
    head = i
    tail = (i+2)%10
    G.add_edge(head,tail,color='g')
    G[head][tail].update(weight=4)
G.remove_edge(4,6)
G.remove_edge(5,7)
G.add_edge(1,5,color='b')
G[1][5].update(weight=4)
G.add_edge(2,6,color='b')
G[2][6].update(weight=4)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
# plt.text(-0.2,-0.25, '$ \;$ $\mathcal{C}_{2}^{\eta}=$ '
#                '\\\ $\mathcal{C}_1 \cup \mathcal{G}_{\eta} \cup \mathcal{D}_{\eta}$ ', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)
plt.text(-0.1,-0.1, '$\mathcal{C}_{2}^{\eta}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figure2Bp3.pdf')
# C_1^union

plt.figure(7,(6,6))
G = cycle_union_Erdos_Renyi(10, k=2, c=0, seed=None,
                            color_the_edges=True,
                            cycle_edge_color='k',
                            weight_the_edges=True,
                            cycle_edge_weights=8)
G.add_edge(1,5,color='b')
G[1][5].update(weight=4)
G.add_edge(2,6,color='b')
G[2][6].update(weight=4)
G.add_edge(4,7,color='b')
G[4][7].update(weight=4)
G.add_edge(3,7,color='b')
G[3][7].update(weight=4)
G.add_edge(2,9,color='b')
G[2][9].update(weight=4)
G.add_edge(5,8,color='b')
G[5][8].update(weight=4)
G.add_edge(4,8,color='b')
G[4][8].update(weight=4)
G.add_edge(1,8,color='b')
G[1][8].update(weight=4)
colors = [G[u][v]['color'] for u,v in G.edges]
weights = [G[u][v]['weight'] for u,v in G.edges]
NX.draw_circular(G, edge_color=colors, width=weights)
# plt.text(-0.15,0,'$\mathcal{C}_{1} \\cup \mathcal{G}_{n,2/n}$', fontsize=30,
#          fontweight='bold', fontdict=None, withdash=False)
plt.text(-0.25,-0.1,'$\mathcal{C}_{1,2/n}$', fontsize=50,
         fontweight='bold', fontdict=None, withdash=False)
plt.savefig('./figures/figure2Bp4.pdf')