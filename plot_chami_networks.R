library(igraph)
#8, 15
el <- read.csv("data/chami-advice-data/edgelists/chami_advice_edgelist_8.txt",
               header = FALSE)

g <- graph_from_edgelist(as.matrix(el), directed = FALSE)
g <- simplify(g)


l1 = layout_with_kk(g)

plot(
    g,
    vertex.size = 2,
    vertex.label = NA, vertex.frame.color = NA,
    vertex.color="black",
    layout = l1
)

