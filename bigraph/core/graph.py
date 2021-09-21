
class Graph:

    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        is_directed=False,
        graph=None
    ):
        import networkx

        if isinstance(nodes, networkx.Graph):
            # `Graph(nx_graph)` -> `graph`
            graph = nodes
            nodes = None
            if edges is not None:
                raise ValueError(
                    "edges: expected no value when using legacy NetworkX constructor, found: {edges!r}"
                )

        # legacy NetworkX construction
        if graph is not None:
            if nodes is not None or edges is not None:
                raise ValueError(
                    "graph: expected no value when using 'nodes' and 'edges' parameters, found: {graph!r}"
                )




