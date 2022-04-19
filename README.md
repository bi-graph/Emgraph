<h1><b>Emgraph</b></h1>
<div>

[//]: # (<a href="https://badge.fury.io/py/emgraph"><img src="https://badge.fury.io/py/emgraph.svg" alt="PyPI version" height="18"></a>)
[//]: # (<a href="https://www.codacy.com/gh/bi-graph/emgraph/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bi-graph/emgraph&amp;utm_campaign=Badge_Grade"><img src="https://app.codacy.com/project/badge/Grade/e320ed8c06a3466aa9711a138085b9d2" alt="PyPI version" height="18"></a>)
[//]: # (<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/emgraph">)

[comment]: <> (<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dw/emgraph">)

[comment]: <> (<img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/emgraph">)

[comment]: <> (<img alt="GitHub search hit counter" src="https://img.shields.io/github/search/bi-graph/emgraph/hit">)

[comment]: <> (<img alt="GitHub search hit counter" src="https://img.shields.io/github/search/bi-graph/emgraph/goto">)

[comment]: <> (<img alt="PyPI - Implementation" src="https://img.shields.io/pypi/implementation/emgraph">)

[comment]: <> (<img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/bi-graph/emgraph">)

[comment]: <> (<img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/bi-graph/emgraph">)
<img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/bi-graph/emgraph?style=social">
</div>
<p><b>Emgraph</b> is a Python toolkit for graph embedding.</p>


[//]: # (<ul>)

[//]: # (    <li><b>Bug reports:</b> https://github.com/bi-graph/emgraph/issues</li>)

[//]: # (</ul>)

[//]: # (> Node based similarities and Katz has been implemented. you can find algorithms in emgraph module. Algorithms implemented so far:)

<div align="center">
<table>
<caption><b>Algorithms table</b></caption>
    <tr>
        <td><b>Number</b></td>
        <td align="center"><b>Algorithm</b></td>
    </tr>
    <tr>
        <td align="center">1</td>
        <td><code><b>TransE</b></code></td>
    </tr>
    <tr>
        <td align="center">2</td>
        <td><code><b>ComplEx</b></code></td>
    </tr>
    <tr>
        <td align="center">3</td>
        <td><code><b>HolE</b></code></td>
    </tr>
    <tr>
        <td align="center">4</td>
        <td><code><b>DistMult</b></code></td>
    </tr>
    <tr>
        <td align="center">5</td>
        <td><code><b>ConvE</b></code></td>
    </tr>
    <tr>
        <td align="center">6</td>
        <td><code><b>ConvKB</b></code></td>
    </tr>
    <tr>
        <td align="center">7</td>
        <td><code><b>RandomBaseline</b></code></td>
    </tr>
</table>
</div>

<div>
  <h2>Installation</h2>
  <p>Install the latest version of <b>Emgraph</b>:</p>

[//]: # (  <pre>$ pip install emgraph</pre>)
  <pre>(will be built soon)</pre>
</div>

<div>
<h2>Documentation</h2>
<p>Soon</p>

[//]: # (<p> <a href="https://emgraph.readthedocs.io/en/latest/index.html">https://emgraph.readthedocs.io/en/latest/</a></p>)

</div>

<h2>Simple example</h2>
<p>Embedding wordnet11 graph using 
<code><b>TransE</b></code> model:</p>

```python
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit
from emgraph.datasets import load_wn11
from emgraph.models import TransE


def train_transe():
    X = load_wn11()
    model = TransE(batches_count=64, seed=0, epochs=20, k=100, eta=20,
                   optimizer='adam', optimizer_params={'lr': 0.0001},
                   loss='pairwise', verbose=True, large_graphs=False)

    model.fit(X['train'])

    scores = model.predict(X['test'])

    print("Scores: ", scores)
    print("Brier score loss:", brier_score_loss(X['test_labels'], expit(scores)))


# Executing the function

if __name__ == '__main__':
    train_transe()
```

<p>Evaluating <code><b>ComplEx</b></code> model after training:<br>

```python
import numpy as np
from emgraph.datasets import load_wn18
from emgraph.models import ComplEx
from emgraph.evaluation import evaluate_performance


def complex_performance():
    X = load_wn18()
    model = ComplEx(batches_count=10, seed=0, epochs=20, k=150, eta=1,
                    loss='nll', optimizer='adam')
    model.fit(np.concatenate((X['train'], X['valid'])))
    filter_triples = np.concatenate((X['train'], X['valid'], X['test']))
    ranks = evaluate_performance(X['test'][:5], model=model,
                                 filter_triples=filter_triples,
                                 corrupt_side='s+o',
                                 use_default_protocol=False)
    return ranks


# Executing the function

if __name__ == '__main__':
    ranks = complex_performance()
    print("ranks {}".format(ranks))
```

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
[//]: # (<div>)

[//]: # (<table>)

[//]: # (<caption><b>Metrics table</b></caption>)

[//]: # (    <tr>)

[//]: # (        <td><b>Number</b></td>)

[//]: # (        <td align="center"><b>Evaluattion metrics</b></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">1</td>)

[//]: # (        <td><code>Precision</code></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">2</td>)

[//]: # (        <td><code>AUC</code></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">3</td>)

[//]: # (        <td><code>ROC</code></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">4</td>)

[//]: # (        <td><code>returns fpr*</code></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">5</td>)

[//]: # (        <td><code>returns tpr*</code></td>)

[//]: # (    </tr>)

[//]: # (</table>)

[//]: # (</div>)

[//]: # ()
[//]: # (> * For further usages and calculating different metrics)

[//]: # ()
[//]: # (<h3>Dataset format</h3>)

[//]: # (<p>Your dataset should be in the following format &#40;Exclude the 'Row' column&#41;:</p>)

[//]: # ()
[//]: # (<div>)

[//]: # (<table>)

[//]: # (<caption><b>Sample edges &#40;links&#41; dataset</b></caption>)

[//]: # (    <tr>)

[//]: # (        <td><b>Row</b></td>)

[//]: # (        <td align="center"><b>left_side</b></td>)

[//]: # (        <td align="center"><b>right_side</b></td>)

[//]: # (        <td align="center"><b>Weight*</b></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">1</td>)

[//]: # (        <td><code>u0</code></td>)

[//]: # (        <td><code>v1</code></td>)

[//]: # (        <td>1</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">2</td>)

[//]: # (        <td><code>u2</code></td>)

[//]: # (        <td><code>v1</code></td>)

[//]: # (        <td>1</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">3</td>)

[//]: # (        <td><code>u1</code></td>)

[//]: # (        <td><code>v2</code></td>)

[//]: # (        <td>1</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">4</td>)

[//]: # (        <td><code>u3</code></td>)

[//]: # (        <td><code>v3</code></td>)

[//]: # (        <td>1</td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">5</td>)

[//]: # (        <td><code>u4</code></td>)

[//]: # (        <td><code>v3</code></td>)

[//]: # (        <td>2</td>)

[//]: # (    </tr>)

[//]: # (</table>)

[//]: # (</div>)

[//]: # ()
[//]: # (> * Note that running <pre>)

[//]: # (    <code>from bigraph.preprocessing import import_files df, df_nodes = import_files&#40;&#41;</code></pre>will create a sample graph for you and will place it in the)

[//]: # (    <code>inputs</code> directory.)

[//]: # (> * Although the weight has not been involved in current version, but, the format will be the same.)

<h3>More examples</h3>
<p>Embedding wordnet11 graph using 
<code><b>DistMult</b></code> model:</p>

```python
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit
from emgraph.datasets import load_wn11
from emgraph.models import DistMult


def train_dist_mult():
    X = load_wn11()
    model = DistMult(batches_count=1, seed=555, epochs=20, k=10, loss='pairwise',
                 loss_params={'margin': 5})

    model.fit(X['train'])

    scores = model.predict(X['test'])

    print("Scores: ", scores)
    print("Brier score loss:", brier_score_loss(X['test_labels'], expit(scores)))


# Executing the function

if __name__ == '__main__':
    train_dist_mult()

```

[//]: # (<h3>References</h3>)

[//]: # (<div>)

[//]: # (<table>)

[//]: # (<caption><b>References table</b></caption>)

[//]: # (    <tr>)

[//]: # (        <td><b>Number</b></td>)

[//]: # (        <td align="center"><b>Reference</b></td>)

[//]: # (        <td align="center"><b>Year</b></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">1</td>)

[//]: # (        <td><code>Yang, Y., Lichtenwalter, R.N. & Chawla, N.V. Evaluating link prediction methods. Knowl Inf Syst 45, 751–782 &#40;2015&#41;.</code> <a href="https://doi.org/10.1007/s10115-014-0789-0")

[//]: # (target="_blank">https://doi.org/10.1007/s10115-014-0789-0</a></td>)

[//]: # (        <td align="center"><b>2015</b></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">2</td>)

[//]: # (        <td><code>Liben-nowell, David & Kleinberg, Jon. &#40;2003&#41;. The Link Prediction Problem for Social Networks. Journal of the American Society for Information Science and Technology.</code><a href="https://doi.org/58.10.1002/asi.20591")

[//]: # (target="_blank">https://doi.org/58.10.1002/asi.20591</a></td>)

[//]: # (        <td align="center"><b>2003</b></td>)

[//]: # (    </tr>)

[//]: # (    <tr>)

[//]: # (        <td align="center">2</td>)

[//]: # (        <td><code>...</code></td>)

[//]: # (        <td align="center"><b>...</b></td>)

[//]: # (    </tr>)

[//]: # (</table>)

[//]: # (</div>)

<h3>Future work</h3>

- [x] Modulate the functions
- [ ] Add more algorithms
- [x] Run on CUDA cores
- [x] Make it faster using vectorization etc.
- [x] Add more preprocessors
- [ ] Add dataset, graph, and dataframe manipulations
- [x] Unify and reconstruct the architecture and eliminate redundancy



<h2>If you found it helpful, please give us a <span>:star:</span></h2>

<h2>License</h3>
<p>Released under the BSD license</p>
<div class="footer"><pre>Copyright &copy; 2019-2022 <b>Emgraph</b> Developers
<a href="https://www.linkedin.com/in/soran-ghaderi/">Soran Ghaderi</a> (soran.gdr.cs@gmail.com)
<a href="https://uk.linkedin.com/in/taleb-zarhesh">Taleb Zarhesh</a> (taleb.zarhesh@gmail.com)</pre>
</div>
