import pygraphviz as pgv
G = pgv.AGraph()
G.add_node("a")
G.add_edge("b", "c")
print(G)
# strict graph "" {
#     a;
#     b - - c
# }
