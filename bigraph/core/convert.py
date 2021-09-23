

def from_networkx(
    graph,
    *,
    node_type_attr,
    edge_type_attr,
    node_type_default,
    edge_type_default,
    edge_weight_attr,
    node_features,
    dtype,
):
    import networkx as nx
