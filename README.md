<div align="center">
<h1><b>Emgraph</b></h1>
<p><b>Emgraph</b> (<b>Em</b>bedding <b>graph</b>s)  is a Python library for graph representation learning.</p>
<p>It provides a simple API for design, train, and evaluate graph embedding models. You can use the base models to easily develop your own model.</p>
</div>

<div align="center">
<a href="https://badge.fury.io/py/emgraph"><img alt="PyPI - Package version" src="https://badge.fury.io/py/emgraph.svg"></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/emgraph">
<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/emgraph">
<img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/emgraph">
<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/bi-graph/emgraph">
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/bi-graph/emgraph?style=social">
<img alt="PyPI - Maintenance" src="https://img.shields.io/badge/Maintained%3F-yes-green.svg">
<img alt="PyPI - License" src="https://img.shields.io/pypi/l/emgraph.svg">
<img alt="PyPI - Format" src="https://img.shields.io/pypi/format/emgraph.svg">
<img alt="Status" src="https://img.shields.io/pypi/status/emgraph.svg">
<img alt="Commits" src="https://badgen.net/github/commits/bi-graph/emgraph">
<img alt="Commits" src="https://img.shields.io/badge/TensorFlow 2-FF6F00?style=flat&logo=tensorflow&logoColor=white">
</div>

<div>
  <h2>Installation</h2>
  <p>Install the latest version of <b>Emgraph</b>:</p>

  <pre>$ pip install emgraph</pre>
</div>

<div>
<h2>Documentation</h2>
<p>Soon</p>

[//]: # (<p> <a href="https://emgraph.readthedocs.io/en/latest/index.html">https://emgraph.readthedocs.io/en/latest/</a></p>)

</div>

<h2>Quick start</h2>
<p>Embedding wordnet11 graph using 
<code><b>TransE</b></code> model:</p>

```python
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit
from emgraph.datasets import BaseDataset, DatasetType
from emgraph.models import TransE


def train_transe(data):
    
    model = TransE(batches_count=64, seed=0, epochs=20, k=100, eta=20,
                   optimizer='adam', optimizer_params={'lr': 0.0001},
                   loss='pairwise', verbose=True, large_graphs=False)
    model.fit(data['train'])
    scores = model.predict(data['test'])
    return scores
    

if __name__ == '__main__':
    
    wn11_dataset = BaseDataset.load_dataset(DatasetType.WN11)
    
    scores = train_transe(data=wn11_dataset)
    print("Scores: ", scores)
    print("Brier score loss:", brier_score_loss(wn11_dataset['test_labels'], expit(scores)))
```

<p>Evaluating <code><b>ComplEx</b></code> model after training:<br>

```python
import numpy as np
from emgraph.datasets import BaseDataset, DatasetType
from emgraph.models import ComplEx
from emgraph.evaluation import evaluate_performance



def complex_performance(data):
    
    model = ComplEx(batches_count=10, seed=0, epochs=20, k=150, eta=1,
                    loss='nll', optimizer='adam')
    model.fit(np.concatenate((data['train'], data['valid'])))
    filter_triples = np.concatenate((data['train'], data['valid'], data['test']))
    ranks = evaluate_performance(data['test'][:5], model=model,
                                 filter_triples=filter_triples,
                                 corrupt_side='s+o',
                                 use_default_protocol=False)
    return ranks


if __name__ == '__main__':

    wn18_dataset = BaseDataset.load_dataset(DatasetType.WN18)   
    ranks = complex_performance(data=wn18_dataset)
    print("ranks {}".format(ranks))
```

<h4>More examples</h4>
<p>Embedding wordnet11 graph using 
<code><b>DistMult</b></code> model:</p>

```python
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit
from emgraph.datasets import BaseDataset, DatasetType
from emgraph.models import DistMult


def train_dist_mult(data):

    model = DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
                     loss_params={'margin': 5})
    model.fit(data['train'])
    scores = model.predict(data['test'])
    
    return scores
    

if __name__ == '__main__':
    
    wn11_dataset = BaseDataset.load_dataset(DatasetType.WN11)
    
    scores = train_dist_mult(data=wn11_dataset)
    print("Scores: ", scores)
    print("Brier score loss:", brier_score_loss(wn11_dataset['test_labels'], expit(scores)))

```

<div align="center">
<table>
<caption><b>Algorithms table</b></caption>
    <tr>
        <td></td>
        <td align="center"><b>Model</b></td>
        <td align="center"><b>Reference</b></td>
    </tr>
    <tr>
        <td align="center">1</td>
        <td><code><b>TransE</b></code></td>
       <td><a href="https://proceedings.neurips.cc/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf">Translating Embeddings for Modeling Multi-relational Data</a></td>
   </tr>
    <tr>
        <td align="center">2</td>
        <td><code><b>ComplEx</b></code></td>
        <td><a href="https://arxiv.org/abs/1606.06357">Complex Embeddings for Simple Link Prediction</a></td>
    </tr>
    <tr>
        <td align="center">3</td>
        <td><code><b>HolE</b></code></td>
        <td><a href="https://arxiv.org/abs/1510.04935">Holographic Embeddings of Knowledge Graphs</a></td>
    </tr>
    <tr>
        <td align="center">4</td>
        <td><code><b>DistMult</b></code></td>
        <td><a href="https://arxiv.org/abs/1412.6575">Embedding Entities and Relations for Learning and Inference in Knowledge Bases</a></td>
    </tr>
    <tr>
        <td align="center">5</td>
        <td><code><b>ConvE</b></code></td>
        <td><a href="https://arxiv.org/abs/1707.01476">Convolutional 2D Knowledge Graph Embeddings</a></td>
    </tr>
    <tr>
        <td align="center">6</td>
        <td><code><b>ConvKB</b></code></td>
        <td><a href="https://arxiv.org/abs/1707.01476">A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network</a></td>
    </tr>    
</table>
</div>

<div>
<h2>Call for Contributions</h2>
<p>The <b>Emgraph</b> project welcomes your expertise and enthusiasm!</p>

<p>Ways to contribute to <b>Emgraph</b>:</p>
<ul>
  <li>Writing code</li>
  <li>Review pull requests</li>
  <li>Develop tutorials, presentations, and other educational materials</li>
  <li>Translate documentation and readme contents</li>
</ul>
</div>

<div>
  <h2>Issues</h2>
  <p>If you happened to encounter any issue in the codes, please report it
    <a href="https://github.com/bi-graph/emgraph/issues">here</a>. 
    A better way is to fork the repository on <b>Github</b> and/or create a pull request.</p>

</div>


[//]: # (<h3>Metrics</h3>)

[//]: # (<p>Metrics that are calculated during evaluation:</p>)

[//]: # ()

[//]: # (> * For further usages and calculating different metrics)

[//]: # ()

[//]: # (<h3>Dataset format</h3>)

[//]: # (<p>Your dataset should be in the following format &#40;Exclude the 'Row' column&#41;:</p>)



<div>
<h3>Features</h3>

- [x] Support CPU/GPU
- [x] Vectorized operations
- [x] Preprocessors
- [x] Dataset loader
- [x] Standard API
- [x] Documentation
- [x] Test driven development
</div>
<h2>If you find it helpful, please give us a <span>:star:</span></h2>

<div>
<h2>License</h2>
<p>Released under the BSD license</p>
</div>

<div class="footer"><pre>Copyright &copy; 2019-2022 <b>Emgraph</b> Developers
<a href="https://soran-ghaderi.github.io/">Soran Ghaderi</a> (soran.gdr.cs@gmail.com)   follow me <a href="https://github.com/soran-ghaderi"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/soranghadri"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/soran-ghaderi/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
<a href="https://uk.linkedin.com/in/taleb-zarhesh">Taleb Zarhesh</a> (taleb.zarhesh@gmail.com)  follow me <a href="https://github.com/sigma1326"><img alt="Github" src="https://img.shields.io/badge/GitHub-100000?&logo=github&logoColor=white"></a> <a href="https://twitter.com/taleb__z"><img alt="Twitter" src="https://img.shields.io/badge/Twitter-1DA1F2?&logo=twitter&logoColor=white"></a> <a href="https://www.linkedin.com/in/taleb-zarhesh/"><img alt="Linkedin" src="https://img.shields.io/badge/LinkedIn-0077B5?&logo=linkedin&logoColor=white"></a>
</pre>
</div>