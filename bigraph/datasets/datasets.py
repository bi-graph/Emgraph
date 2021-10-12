import pandas as pd
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def load_from_csv(directory_path, file_name, sep='\t', header=None, add_reciprocal_rels=False):
    """
    Load data from a CSV file as:
    ... code-block:: text

       subj1    relation1   obj1
       subj1    relation2   obj2
       subj3    relation3   obj2
       subj4    relation1   obj2
   ...

    :param directory_path: Input directory path
    :type directory_path: str
    :param file_name: File name
    :type file_name: str
    :param sep: Seperator to split columns by (default \t)
    :type sep: str
    :param header: CSV file header row (like in pandas)
    :type header: int, None
    :param add_reciprocal_rels: Flag which specifies whether to add reciprocal relations. For every <s, p, o> in the
    dataset this creates a corresponding triple with reciprocal relation <o, p_reciprocal, s>. (default: False)
    :type add_reciprocal_rels: bool
    :return: The actual triples of the file
    :rtype: ndarray , shape [n, 3]
    """

    logger.debug('Loading data from {}.'.format(file_name))
    df = pd.read_csv(os.path.join(directory_path, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    if add_reciprocal_rels:
        df = _add_reciprocal_relations(df)

    return df.values


def _add_reciprocal_relations(tri_df):
    """
    Add reciprocal relations to the triples.

    :param tri_df: Dataframe of triples
    :type tri_df: Dataframe
    :return: Dataframe of triples and their reciprocals
    :rtype: Dataframe
    """
    df_reciprocal = tri_df.copy()
    # swap subjects and objects
    cols = list(df_reciprocal.columns)
    cols[0], cols[2] = cols[2], cols[0]
    df_reciprocal.columns = cols
    # add reciprocal relations
    df_reciprocal.iloc[:, 1] = df_reciprocal.iloc[:, 1] + "_reciprocal"
    # append to original triples
    tri_df = tri_df.append(df_reciprocal)
    return tri_df
