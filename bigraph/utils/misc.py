import numpy as np
import logging

SUBJECT = 0
PREDICATE = 1
OBJECT = 2
DEBUG = True

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def get_entity_triples(entity, graph):
    """
    Return all triples of the `graph` where `entity` is a part of.

    :param entity: Entity label
    :type entity: str, shape [n, 1]
    :param graph: ND. array of triples
    :type graph: np.ndarray, shape [n, 3]
    :return: ND. array of the triples containing `entity`
    :rtype: np.ndarray, shape [n, 3]
    """

    logger.debug('Return a list of all triples where {} appears as subject or object.'.format(entity))
    # NOTE: The current implementation is slightly faster (~15%) than the more readable one-liner:
    #           rows, _ = np.where((entity == graph[:,[SUBJECT,OBJECT]]))

    # Get rows and cols where entity is found in graph
    rows, cols = np.where((entity == graph))

    # In the unlikely event that entity is found in the relation column (index 1)
    rows = rows[np.where(cols != PREDICATE)]

    # Subset graph to neighbourhood of entity
    neighbours = graph[rows, :]

    return neighbours
