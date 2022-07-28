import enum
import hashlib
import logging
import mimetypes
import os
import shutil
import typing
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pydantic

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

Emgraph_ENV_NAME = "Emgraph_DATA_HOME"


class DatasetType(enum.Enum):
    UNKNOWN = 0
    WN11 = 1
    WN18 = 2
    WN18RR = 3
    FB13 = 4
    FB15K = 5
    FB15K_237 = 6
    YAGO3_10 = 7
    ONET20K = 8
    PPI5K = 9
    NL27K = 10
    CN15K = 11


class File(pydantic.BaseModel):
    name: typing.Optional[str]
    checksum: typing.Optional[str]


class BaseDataset(pydantic.BaseModel):
    """
    The base class for dataset classes to implement from.
    All methods for working with different kinds of dataset files are implemented here, dataset classes that derive from this base class only need to
    implement `after_load` and `after_clean` methods if that step is needed by that dataset.

    """

    _registry: typing.Dict[DatasetType, "BaseDataset"] = {}
    _mime = mimetypes.MimeTypes()

    name: typing.Optional[str]
    type: typing.Optional[DatasetType]
    file_name: typing.Optional[str]
    url: typing.Optional[str]

    train_file: typing.Optional[File]
    test_file: typing.Optional[File]
    valid_file: typing.Optional[File]

    clean_unseen: typing.Optional[bool] = False
    generate_focusE: typing.Optional[bool] = False
    split_test_into_top_bottom: typing.Optional[bool] = False

    @classmethod
    def after_load(
        cls,
        dataset: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """
        This method is called after the dataset has been loaded.

        :param dataset: The dataset before processing
        :type dataset: dict[str, np.ndarray]
        :return: The dataset after processing
        :rtype: dict[str, np.ndarray]
        """
        return dataset

    @classmethod
    def after_clean(
        cls,
        dataset: dict[str, np.ndarray],
        filtered_dataset: dict[str, np.ndarray],
        valid_idx: object,
        test_idx: object,
    ) -> dict[str, np.ndarray]:
        """
        This method is called when `clean_dataset` parameter is set to true.

        :param dataset: The dataset before cleaning
        :type dataset: dict[str, np.ndarray]
        :param filtered_dataset:
        :type filtered_dataset: dict[str, np.ndarray]
        :param valid_idx: The valid indices of the validation dataset
        :type valid_idx: np.array
        :param test_idx: The valid indices of the test dataset.
        :type test_idx: np.array
        :rtype: dict[str, np.ndarray]
        :return: The dataset after cleaning
        """
        pass

    @classmethod
    def __init_subclass__(cls, **kwargs):
        _temp = cls()
        BaseDataset._registry[_temp.type] = _temp

    @staticmethod
    def _md5(file_path: str) -> str:
        """
        Calculates the md5 hash of a file with the given path.

        :param file_path: The path of the file
        :type file_path: str
        :return: The calculated hash of the file
        :rtype: str
        """
        md5hash = hashlib.md5()
        chunk_size = 4096
        with open(file_path, "rb") as f:
            content_buffer = f.read(chunk_size)
            while content_buffer:
                md5hash.update(content_buffer)
                content_buffer = f.read(chunk_size)
        return md5hash.hexdigest()

    @classmethod
    def _add_reciprocal_relations(
        cls,
        tri_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add reciprocal relations to the triples.

        :param tri_df: Dataframe of triples
        :type tri_df: pd.Dataframe
        :return: Dataframe of triples and their reciprocals
        :rtype: pd.Dataframe
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

    @classmethod
    def _unzip_dataset(
        cls,
        dataset: "BaseDataset",
        source: str,
        destination: str,
        check_md5hash: bool = False,
    ) -> None:
        """
        Unzips a dataset with given source path and extracts into the destination path.
        The source file of the dataset is removed after extraction.

        :param dataset: The dataset instance to use for extraction
        :type dataset: BaseDataset
        :param source: Path of the dataset to extract
        :type source: str
        :param destination: Path to extract the dataset into
        :type destination: str
        :param check_md5hash: Whether to check MD5 hash for the dataset files or not
        :type check_md5hash: bool
        :rtype: None
        """
        # TODO - add error checking
        try:
            shutil.unpack_archive(source, destination)
        except Exception as e:
            logger.exception(e)
        else:
            if check_md5hash:
                for file in [
                    dataset.train_file,
                    dataset.test_file,
                    dataset.valid_file,
                ]:
                    file_path = os.path.join(destination, dataset.name, file.name)
                    checksum = cls._md5(file_path)
                    if checksum != file.checksum:
                        os.remove(source)
                        msg = (
                            "{} has an md5 checksum of ({}) which is different from the expected ({}), "
                            "the file may be corrupted.".format(
                                file_path, checksum, file.checksum
                            )
                        )
                        logger.error(msg)
                        raise IOError(msg)
            os.remove(source)

    @classmethod
    def _fetch_remote_data(
        cls,
        remote: "BaseDataset",
        download_dir: str,
        data_home: str,
        check_md5hash: bool = False,
    ) -> None:
        """
        Fetches the dataset files from a remote url and extracts it.

        :param remote: The dataset to get from the remote url
        :type remote: BaseDataset
        :param download_dir: The path to save downloaded dataset into
        :type download_dir: str
        :param data_home: The path where all Emgraph datasets are stored
        :type data_home: str
        :param check_md5hash: Whether to check MD5 hash for the dataset files or not
        :type check_md5hash: bool
        """
        # get from remote
        # self._mime.guess_type(url)

        file_path = "{}.zip".format(download_dir)
        if not Path(file_path).exists():
            urllib.request.urlretrieve(remote.url, file_path)
            # TODO - add error checking
        cls._unzip_dataset(remote, file_path, data_home, check_md5hash)

    @classmethod
    def _get_data_home(
        cls,
        data_home: typing.Optional[str] = None,
    ) -> str:
        """
        Gets the path where all Emgraph datasets are stored and creates the path if it does not exist.

        :param data_home: The path where all Emgraph datasets are stored
        :type data_home: str
        :return: The path where all Emgraph datasets are stored
        :rtype: str
        """

        if data_home is None:
            data_home = os.environ.get(
                Emgraph_ENV_NAME,
                os.path.join(
                    os.getcwd(),
                    "emgraph_datasets",
                ),
            )

        data_home = os.path.expanduser(data_home)
        if not os.path.exists(data_home):
            os.makedirs(data_home)
        logger.debug("data_home is set to {}".format(data_home))
        return data_home

    @classmethod
    def load_from_ntriples(
        cls,
        folder_name: str,
        file_name: str,
        data_home: typing.Optional[str] = None,
        add_reciprocal_rels: bool = False,
    ):
        logger.debug("Loading rdf ntriples from {}.".format(file_name))
        data_home = cls._get_data_home(data_home)
        df = pd.read_csv(
            os.path.join(data_home, folder_name, file_name),
            sep=r"\s+",
            header=None,
            names=None,
            dtype=str,
            usecols=[0, 1, 2],
        )

        # Remove trailing full stop (if present)
        df[2] = df[2].apply(lambda x: x.rsplit(".", 1)[0])

        if add_reciprocal_rels:
            df = cls._add_reciprocal_relations(df)

        return df.values

    @classmethod
    def fetch_dataset(
        cls,
        dataset: "BaseDataset",
        data_home: typing.Optional[str] = None,
        check_md5hash: bool = False,
    ):
        """
         Fetches the dataset files from a remote url and extracts it.
        
        :param dataset: The dataset to fetch
        :type dataset: BaseDataset
        :param data_home: The path where all Emgraph datasets are stored
        :type data_home: str
        :param check_md5hash: Whether to check MD5 hash for the dataset files or not
        :type check_md5hash: bool
        :return: Path where the given dataset has been saved
        :rtype: str
        """
        data_home = cls._get_data_home(data_home)
        dataset_dir = os.path.join(data_home, dataset.name)
        if not os.path.exists(dataset_dir):
            if dataset.url is None:
                msg = "No dataset at {} and no url provided.".format(dataset_dir)
                logger.error(msg)
                raise Exception(msg)

            cls._fetch_remote_data(dataset, dataset_dir, data_home, check_md5hash)
        return dataset_dir

    # todo: try this as well for the datasets: https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/data/datasets.py
    # todo: add generators: https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/data/generator.py
    # todo: use this structure for the core graph class: https://github.com/Sujit-O/pykg2vec/blob/master/pykg2vec/data/kgcontroller.py
    @classmethod
    def _clean_data(cls, X, return_idx=False):
        if X["train"].shape[1] == 3:
            columns = ["s", "p", "o"]
        else:
            columns = ["s", "p", "o", "w"]

        train = pd.DataFrame(X["train"], columns=columns)
        valid = pd.DataFrame(X["valid"], columns=columns)
        test = pd.DataFrame(X["test"], columns=columns)

        train_ent = np.unique(np.concatenate((train.s, train.o)))
        train_rel = train.p.unique()

        valid_idx = (
            valid.s.isin(train_ent) & valid.o.isin(train_ent) & valid.p.isin(train_rel)
        )
        test_idx = (
            test.s.isin(train_ent) & test.o.isin(train_ent) & test.p.isin(train_rel)
        )

        filtered_valid = valid[valid_idx].values
        filtered_test = test[test_idx].values

        filtered_X = {
            "train": train.values,
            "valid": filtered_valid,
            "test": filtered_test,
        }

        cls.after_clean(X, filtered_X, valid_idx, test_idx)

        if return_idx:
            return filtered_X, valid_idx, test_idx
        else:
            return filtered_X

    @classmethod
    def load_from_csv(
        cls,
        directory_path: str,
        file_name: str,
        sep: str = "\t",
        header: typing.Optional[str] = None,
        add_reciprocal_rels: bool = False,
    ) -> np.ndarray:
        """
        Load data from a CSV file as:
        .. code-block:: text

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
        :rtype: np.ndarray , shape [n, 3]
        """

        logger.debug("Loading data from {}.".format(file_name))
        df = pd.read_csv(
            os.path.join(directory_path, file_name),
            sep=sep,
            header=header,
            names=None,
            dtype=str,
        )
        logger.debug("Dropping duplicates.")
        df = df.drop_duplicates()
        if add_reciprocal_rels:
            df = cls._add_reciprocal_relations(df)

        return df.values

    @classmethod
    def load_from_rdf(
        cls,
        folder_name: str,
        file_name: str,
        rdf_format: str = "nt",
        data_home: typing.Optional[str] = None,
        add_reciprocal_rels: bool = False,
    ) -> np.ndarray:
        """
        Loads from a RDF file format.

        :param folder_name:
        :type folder_name: str
        :param file_name:
        :type file_name: str
        :param rdf_format:
        :type rdf_format: str
        :param data_home:
        :type data_home: str
        :param add_reciprocal_rels:
        :type add_reciprocal_rels:
        :return:
        :rtype: np.ndarray
        """
        logger.debug("Loading rdf data from {}.".format(file_name))
        data_home = BaseDataset._get_data_home(data_home)
        from rdflib import Graph

        g = Graph()
        g.parse(
            os.path.join(data_home, folder_name, file_name),
            format=rdf_format,
            publicID="http://test#",
        )
        triples = pd.DataFrame(np.array(g))
        triples = triples.drop_duplicates()
        if add_reciprocal_rels:
            triples = cls._add_reciprocal_relations(triples)

        return triples.values

    @classmethod
    def generate_focusE_dataset_splits(
        cls,
        dataset: dict[str, np.ndarray],
        split_test_into_top_bottom: bool = True,
        split_threshold: float = 0.1,
    ):
        dataset["train_numeric_values"] = dataset["train"][:, 3].astype(np.float32)
        dataset["valid_numeric_values"] = dataset["valid"][:, 3].astype(np.float32)
        dataset["test_numeric_values"] = dataset["test"][:, 3].astype(np.float32)

        dataset["train"] = dataset["train"][:, 0:3]
        dataset["valid"] = dataset["valid"][:, 0:3]
        dataset["test"] = dataset["test"][:, 0:3]

        sorted_indices = np.argsort(dataset["test_numeric_values"])
        dataset["test"] = dataset["test"][sorted_indices]
        dataset["test_numeric_values"] = dataset["test_numeric_values"][sorted_indices]

        if split_test_into_top_bottom:
            split_threshold = int(split_threshold * dataset["test"].shape[0])

            dataset["test_bottomk"] = dataset["test"][:split_threshold]
            dataset["test_bottomk_numeric_values"] = dataset["test_numeric_values"][
                :split_threshold
            ]

            dataset["test_topk"] = dataset["test"][-split_threshold:]
            dataset["test_topk_numeric_values"] = dataset["test_numeric_values"][
                -split_threshold:
            ]

        return dataset

    @staticmethod
    def load_dataset(
        dataset_type: DatasetType,
        clean_unseen=None,
        data_home=None,
        check_md5hash=False,
        add_reciprocal_rels=False,
        split_threshold=0.1,
    ) -> typing.Optional[dict[str, typing.Any]]:
        dataset = BaseDataset._registry.get(dataset_type, None)
        if dataset is None:
            return None

        if clean_unseen is not None:
            dataset.clean_unseen = clean_unseen

        if dataset.url is None:
            raise ValueError(
                "The dataset name or url must be provided to load a dataset."
            )

        dataset_dir = dataset.fetch_dataset(dataset, data_home, check_md5hash)
        train = dataset.load_from_csv(
            dataset_dir,
            dataset.train_file.name,
            add_reciprocal_rels=add_reciprocal_rels,
        )
        valid = dataset.load_from_csv(
            dataset_dir,
            dataset.valid_file.name,
            add_reciprocal_rels=add_reciprocal_rels,
        )
        test = dataset.load_from_csv(
            dataset_dir,
            dataset.test_file.name,
            add_reciprocal_rels=add_reciprocal_rels,
        )

        _temp = {"train": train, "valid": valid, "test": test}
        _temp = dataset.after_load(_temp)
        if dataset.clean_unseen:
            _temp = dataset._clean_data(_temp)

        if dataset.generate_focusE:
            _temp = dataset.generate_focusE_dataset_splits(
                _temp, dataset.split_test_into_top_bottom, split_threshold
            )

        return _temp


class WN11(BaseDataset):
    name = "wordnet11"
    type = DatasetType.WN11
    file_name = "wordnet11.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/wordnet11.zip"

    train_file = File(name="train.txt", checksum="2429c672c89e33ad4fa8e1a3ade416e4")
    test_file = File(name="test.txt", checksum="24113b464f8042c339e3e6833c1cebdf")
    valid_file = File(name="dev.txt", checksum="87bf86e225e79294a2524089614b96aa")

    clean_unseen = True
    generate_focusE = False
    split_test_into_top_bottom = False

    @classmethod
    def after_load(cls, dataset):
        valid_labels = dataset["valid"][:, 3]
        test_labels = dataset["test"][:, 3]

        dataset["valid"] = dataset["valid"][:, 0:3]
        dataset["test"] = dataset["test"][:, 0:3]

        dataset["valid_labels"] = valid_labels == "1"
        dataset["test_labels"] = test_labels == "1"

        return dataset

    @classmethod
    def after_clean(cls, dataset, filtered_dataset, valid_idx, test_idx):
        filtered_dataset["valid_labels"] = dataset["valid_labels"][valid_idx]
        filtered_dataset["test_labels"] = dataset["test_labels"][test_idx]


class WN18(BaseDataset):
    name = "wn18"
    type = DatasetType.WN18
    file_name = "wn18.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/wn18.zip"

    train_file = File(name="train.txt", checksum="7d68324d293837ac165c3441a6c8b0eb")
    test_file = File(name="test.txt", checksum="b035247a8916c7ec3443fa949e1ff02c")
    valid_file = File(name="valid.txt", checksum="f4f66fec0ca83b5ebe7ad7003404e61d")

    clean_unseen = False
    generate_focusE = False
    split_test_into_top_bottom = False


class WN18RR(BaseDataset):
    name = "wn18RR"
    type = DatasetType.WN18RR
    file_name = "wn18RR.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/wn18RR.zip"

    train_file = File(name="train.txt", checksum="35e81af3ae233327c52a87f23b30ad3c")
    test_file = File(name="test.txt", checksum="2b45ba1ba436b9d4ff27f1d3511224c9")
    valid_file = File(name="valid.txt", checksum="74a2ee9eca9a8d31f1a7d4d95b5e0887")

    clean_unseen = True
    generate_focusE = False
    split_test_into_top_bottom = False


class FB13(BaseDataset):
    name = "freebase13"
    type = DatasetType.FB13
    file_name = "freebase13.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/freebase13.zip"

    train_file = File(name="train.txt", checksum="9099ebcd85ab3ce723cfaaf34f74dceb")
    test_file = File(name="test.txt", checksum="f9af2eac7c5a86996c909bdffd295528")
    valid_file = File(name="dev.txt", checksum="c4ef7b244baa436a97c2a5e57d4ba7ed")

    clean_unseen = True

    # fixme : add a custom dataset loader
    @classmethod
    def after_load(cls, dataset):
        valid_labels = dataset["valid"][:, 3]
        test_labels = dataset["test"][:, 3]

        dataset["valid"] = dataset["valid"][:, 0:3]
        dataset["test"] = dataset["test"][:, 0:3]

        dataset["valid_labels"] = valid_labels == "1"
        dataset["test_labels"] = test_labels == "1"

        return dataset

    @classmethod
    def after_clean(cls, dataset, filtered_dataset, valid_idx, test_idx):
        filtered_dataset["valid_labels"] = dataset["valid_labels"][valid_idx]
        filtered_dataset["test_labels"] = dataset["test_labels"][test_idx]


class FB15K(BaseDataset):
    name = "fb15k"
    type = DatasetType.FB15K
    file_name = "fb15k.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/fb15k.zip"

    train_file = File(name="train.txt", checksum="5a87195e68d7797af00e137a7f6929f2")
    test_file = File(name="test.txt", checksum="71098693b0efcfb8ac6cd61cf3a3b505")
    valid_file = File(name="valid.txt", checksum="275835062bb86a86477a3c402d20b814")

    clean_unseen = False
    generate_focusE = False
    split_test_into_top_bottom = False


class FB15K_237(BaseDataset):
    name = "fb15k-237"
    type = DatasetType.FB15K_237
    file_name = "fb15k-237.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/fb15k-237.zip"

    train_file = File(name="train.txt", checksum="c05b87b9ac00f41901e016a2092d7837")
    test_file = File(name="test.txt", checksum="f5bdf63db39f455dec0ed259bb6f8628")
    valid_file = File(name="valid.txt", checksum="6a94efd530e5f43fcf84f50bc6d37b69")

    clean_unseen = True
    generate_focusE = False
    split_test_into_top_bottom = False


class YAGO3_10(BaseDataset):
    name = "YAGO3-10"
    type = DatasetType.YAGO3_10
    file_name = "YAGO3-10.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/YAGO3-10.zip"

    train_file = File(name="train.txt", checksum="a9da8f583ec3920570eeccf07199229a")
    test_file = File(name="test.txt", checksum="14bf97890b2fee774dbce5f326acd189")
    valid_file = File(name="valid.txt", checksum="2d679a906f2b1ac29d74d5c948c1ad09")

    clean_unseen = True
    generate_focusE = False
    split_test_into_top_bottom = False


class ONET20K(BaseDataset):
    name = "onet20k"
    type = DatasetType.ONET20K
    file_name = "onet20k.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/onet20k.zip"

    train_file = File(name="train.tsv", checksum="516220427a9a18516fd7a804a6944d64")
    test_file = File(name="test.tsv", checksum="e5baec19037cb0bddc5a2fe3c0f4445a")
    valid_file = File(name="valid.tsv", checksum="d7806951ac3d916c5c5a0304eea064d2")

    clean_unseen = True
    split_test_into_top_bottom = True
    generate_focusE = True


class PPI5K(BaseDataset):
    name = "ppi5k"
    type = DatasetType.PPI5K
    file_name = "ppi5k.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/ppi5k.zip"

    train_file = File(name="train.tsv", checksum="d8b54de3482c0d043118cbd05f2666cf")
    test_file = File(name="test.tsv", checksum="7e6e345f496ed9a0cc58b91d4877ddd6")
    valid_file = File(name="valid.tsv", checksum="2bd094118f4be1f4f6d6a1d4707271c1")

    clean_unseen = True
    generate_focusE = True
    split_test_into_top_bottom = True


class NL27K(BaseDataset):
    name = "nl27k"
    type = DatasetType.NL27K
    file_name = "nl27k.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/nl27k.zip"

    train_file = File(name="train.tsv", checksum="d4ce775401d299074d98e046f13e7283")
    test_file = File(name="test.tsv", checksum="2ba17f29119688d93c9d29ab40f63b3e")
    valid_file = File(name="valid.tsv", checksum="00177fa6b9f5cec18814ee599c02eae3")

    clean_unseen = True
    split_test_into_top_bottom = True
    generate_focusE = True


class CN15K(BaseDataset):
    name = "cn15k"
    type = DatasetType.CN15K
    file_name = "cn15k.zip"
    url = "https://github.com/bi-graph/KGdatasets/raw/master/datasets/cn15k.zip"

    train_file = File(name="train.tsv", checksum="8bf2ecc8f34e7b3b544afc30abaac478")
    test_file = File(name="test.tsv", checksum="29df4b8d24a3d89fc7c1032b9c508112")
    valid_file = File(name="valid.tsv", checksum="15b63ebd7428a262ad5fe869cc944208")

    clean_unseen = True
    split_test_into_top_bottom = True
    generate_focusE = True
