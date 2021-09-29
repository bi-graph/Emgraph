import warnings
from collections import defaultdict, namedtuple
from typing import Iterable

import pandas as pd
import numpy as np


def separated(values, *, limit, stringify, sep):
    """
    Print ``limit`` values with a specified seperator.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
        sep (str): the separator to use between elements (and the "... (NNN more)" continuation)
    """
    count = len(values)
    if limit is not None and count > limit:
        values = values[:limit]
        continuation = f"{sep}... ({count - limit} more)" if count > limit else ""
    else:
        continuation = ""

    rendered = sep.join(stringify(x) for x in values)
    return rendered + continuation

def comma_sep(values, limit=20, stringify=repr):
    """
    Print up to ``limit`` values, comma separated.

    Args:
        values (list): the values to print
        limit (optional, int): the maximum number of values to print (None for no limit)
        stringify (callable): a function to use to convert values to strings
    """
    return separated(values, limit=limit, stringify=stringify, sep=", ")

def zero_sized_array(shape, dtype):
    """
    Create an empty array.

    Args:
        shape (tuple): a shape tuple that contains at least one 0

    Returns:
        A NumPy array that contains no elements, and has a small allocation.
    """
    if 0 not in shape:
        raise ValueError("shape: expected at least one zero, found {shape}")

    dtype = np.dtype(dtype)
    return np.broadcast_to(dtype.type(), shape)

def _features_from_attributes(node_type, ids, values, dtype):
    # the size is the first element that has a length, or None if there's only None elements.
    size = next((len(x) for x in values if x is not None), None)

    num_nodes = len(ids)

    if size is None:
        # no features = zero-dimensional features, and skip the loop below
        return zero_sized_array((num_nodes, 0), dtype)

    default_value = np.zeros(size, dtype)

    missing = []

    def compute_value(node_id, x):
        if x is None:
            missing.append(node_id)
            return default_value
        elif len(x) != size:
            raise ValueError(
                f"inferred all nodes of type {node_type!r} to have feature dimension {size}, found dimension {len(x)}"
            )

        return x

    matrix = np.array(
        [compute_value(node_id, x) for node_id, x in zip(ids, values)], dtype
    )
    assert matrix.shape == (num_nodes, size)

    if missing:
        # user code is 5 frames above the warnings.warn call
        stacklevel = 5
        warnings.warn(
            f"found the following nodes (of type {node_type!r}) without features, using {size}-dimensional zero vector: {comma_sep(missing)}",
            stacklevel=stacklevel,
        )

    return matrix

def _features_from_node_data(nodes, node_type_default, data, dtype):
    if isinstance(data, dict):

        def single(node_type):
            node_info = nodes[node_type]
            try:
                this_data = data[node_type]
            except KeyError:
                # no data specified for this type, so len(feature vector) = 0 for each node (this
                # uses a range index for columns, to match the behaviour of the other feature
                # converters here, that build DataFrames from NumPy arrays even when there's no
                # data, i.e. array.shape = (num nodes, 0))
                this_data = pd.DataFrame(columns=range(0), index=node_info.ids)

            if isinstance(this_data, pd.DataFrame):
                df = this_data.astype(dtype, copy=False)
            elif isinstance(this_data, (Iterable, list)):
                # this functionality is a bit peculiar (Pandas is generally nicer), and is
                # undocumented. Consider deprecating and removing it.
                ids, values = zip(*this_data)
                df = pd.DataFrame(values, index=ids, dtype=dtype)
            else:
                raise TypeError(
                    f"node_features[{node_type!r}]: expected DataFrame or iterable, found {type(this_data).__name__}"
                )

            graph_ids = set(node_info.ids)
            data_ids = set(df.index)
            if graph_ids != data_ids:
                parts = []
                missing = graph_ids - data_ids
                if missing:
                    parts.append(f"missing from data ({comma_sep(list(missing))})")
                extra = data_ids - graph_ids
                if extra:
                    parts.append(f"extra in data ({comma_sep(list(extra))})")
                message = " and ".join(parts)
                raise ValueError(
                    f"node_features[{node_type!r}]: expected feature node IDs to exactly match nodes in graph; found: {message}"
                )

            return df

        return {node_type: single(node_type) for node_type in nodes.keys()}

    elif isinstance(data, pd.DataFrame):
        if len(nodes) > 1:
            raise TypeError(
                "When there is more than one node type, pass node features as a dictionary."
            )

        node_type = next(iter(nodes), node_type_default)
        return _features_from_node_data(
            nodes, node_type_default, {node_type: data}, dtype
        )

    elif isinstance(data, (Iterable, list)):
        id_to_data = dict(data)
        return {
            node_type: pd.DataFrame(
                (id_to_data[x] for x in node_info.ids), index=node_info.ids, dtype=dtype
            )
            for node_type, node_info in nodes.items()
        }




SingleTypeNodeIdsAndFeatures = namedtuple(
    "SingleTypeNodeIdsAndFeatures", ["ids", "features"]
)


def _empty_node_info() -> SingleTypeNodeIdsAndFeatures:
    return SingleTypeNodeIdsAndFeatures([], [])

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

    nodes = defaultdict(_empty_node_info)

    features_in_node = isinstance(node_features, str)

    for node_id, node_data in graph.nodes(data=True):
        node_type = node_data.get(node_type_attr, node_type_default)
        node_info = nodes[node_type]
        node_info.ids.append(node_id)
        if features_in_node:
            node_info.features.append(node_data.get(node_features, None))

    if features_in_node or node_features is None:
        node_frames = {
            node_type: pd.DataFrame(
                _features_from_attributes(
                    node_type, node_info.ids, node_info.features, dtype
                ),
                index=node_info.ids,
            )
            for node_type, node_info in nodes.items()
        }
    else:
        node_frames = _features_from_node_data(
            nodes, node_type_default, node_features, dtype
        )