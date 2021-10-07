from . import convert

FEATURE_ATTR_NAME = "feature"
TARGET_ATTR_NAME = "target"
TYPE_ATTR_NAME = "label"
UNKNOWN_TARGET_ATTRIBUTE = "-1"
NODE_TYPE_DEFAULT = "default"
EDGE_TYPE_DEFAULT = "default"

SOURCE = "source"
TARGET = "target"
WEIGHT = "weight"


class Graph:

    def __init__(
        self,
        nodes=None,
        edges=None,
        *,
        is_directed=False,
        source_column=SOURCE,
        target_column=TARGET,
        edge_weight_column=WEIGHT,
        edge_type_column=None,
        node_type_default=NODE_TYPE_DEFAULT,
        edge_type_default=EDGE_TYPE_DEFAULT,
        dtype="float32",
        # legacy arguments:
        graph=None,
        node_type_name=TYPE_ATTR_NAME,
        edge_type_name=TYPE_ATTR_NAME,
        node_features=None,
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

            nodes, edges = convert.from_networkx(
                graph,
                node_type_attr=node_type_name,
                edge_type_attr=edge_type_name,
                node_type_default=node_type_default,
                edge_type_default=edge_type_default,
                edge_weight_attr=edge_weight_column,
                node_features=node_features,
                dtype=dtype,
            )



