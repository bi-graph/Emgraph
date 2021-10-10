import pandas as pd
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def load_from_csv(directory_path, file_name, sep='\t', header=None, add_reciprocal_rels=False):

    logger.debug('Loading data from {}.'.format(file_name))
    df = pd.read_csv(os.path.join(directory_path, file_name),
                     sep=sep,
                     header=header,
                     names=None,
                     dtype=str)
    logger.debug('Dropping duplicates.')
    df = df.drop_duplicates()
    if add_reciprocal_rels:
        df_reciprocal = df.copy()

        # swap subjects and objects
        cols = list(df_reciprocal.columns)
        cols[0], cols[2] = cols[2], cols[0]
        df_reciprocal.columns = cols

        # add reciprocal relations
        df_reciprocal.iloc[:, 1] = df_reciprocal.iloc[:, 1] + "_reciprocal"

        # append to original triples
        triples_df = df.append(df_reciprocal)

    return df.values