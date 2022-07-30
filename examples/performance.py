import os

import numpy as np

from emgraph.datasets import BaseDataset, DatasetType
from emgraph.evaluation import evaluate_performance
from emgraph.models import ComplEx, ConvKB, DistMult, HolE, TransE

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

classes = (ComplEx, HolE, TransE, ConvKB, DistMult)
for cls in classes:
    X = BaseDataset.load_dataset(DatasetType.WN18)
    print(f"model name: {cls.__name__}")

    model = cls(
        batches_count=10, seed=0, epochs=1, k=150, eta=1, loss="nll", optimizer="adam"
    )
    model.fit(np.concatenate((X["train"], X["valid"])))

    filter_triples = np.concatenate((X["train"], X["valid"], X["test"]))
    ranks = evaluate_performance(
        X["test"][:5],
        model=model,
        filter_triples=filter_triples,
        corrupt_side="s+o",
        use_default_protocol=False,
    )

    print(f"ranks: {ranks}\n\n")
