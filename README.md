# *DOC*: Digital Outrage Classifier

> Developed by members of the Crockett Lab at Yale University, `DOC` is a Python package that allows researchers to predict the probability that text input contains moral outrage. 

> The code and materials in this repository are adapted from "[Social reinforcement of moral outrage in online social networks](www.google.com)", published in *TBD* in 2019.

[![made-with-python][made-with-python]](https://www.python.org/)
[![Outrageclf version][outrage-image]](www.google.com)
[![Build Status][travis-image]][travis-url]
[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](www.google.com)

## Installation

The first step is to clone the repo:

```sh
git clone https://github.com/CrockettLab/outrage_classifier
```

then navigate to the folder and install locally with the `setup.py` file. The package is compatible with both Python2 and Python3
```sh
python3 setup.py install
```

or

```sh
python setup.py install
```

## Importing
The package can be imported using the following code:

```python
import outrageclf as oclf
from outrageclf.preprocessing import WordEmbed, get_lemmatize_hashtag
from outrageclf.classifier import _load_crockett_model
```

For those using macOS, a runtime error (described [here](https://stackoverflow.com/questions/53014306/error-15-initializing-libiomp5-dylib-but-found-libiomp5-dylib-already-initial)) may prevent the package from being successfully imported. If you experience this issue, setting the environment varibale `KMP_DUPLICATE_LIB_OK` to `TRUE` should solve the problem:

```python
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
```

## Usage:
The current version of `outrageclf` allows users to predict moral outrage using the pre-trained deep GRU model, which are described in detail in our paper [here](www.google.com). 

To replicate our results or run the pre-trained model, one would need the following files. If interested, please email [to be filled](www.google.com) for additional information.

- [x] A pre-trained embedding model, stored in a `.joblib` format
- [x] A pre-trained GRU model, stored in a `.h5` format 

In general, to predict the level of outrage in a list of texts, we follow the pipelines:

```mermaid
Load pretrained models -> Preprocess text -> Embed text -> Make prediction 
```

The following lines show a complete example of using our pretrained-models. Note that this example assume that you have acquired (or independently trained) both an embedding model stored at `embedding_url` and a classifier stored at `model_url`.

```python

tweets = ["This topic infuriates me because it violates my moral stance", "This is just a super-normal topic #normal"]

# loading our pre-trained models
word_embed = WordEmbed()
word_embed._get_pretrained_tokenizer(embedding_url)
model = _load_crockett_model(model_url)

# the text are lemmatized and embedded into 50-d space
lemmatized_text = get_lemmatize_hashtag(text_vector)
embedded_vector = word_embed._get_embedded_vector(lemmatized_text)
predict = model.predict(embedded_vector)
```

Alternatively, you can also use the model's wrapper function, stored in the `classifier` module. This step bypasses the need to lemmatize and embed the text input:

```python
from outrageclf.classifier import pretrained_model_predict
pretrained_model_predict(tweets, embedding_url, model_url)
```

**Example Notebook:**

There is also an example notebook that has a self-containing example of these two use cases. Refer to `example.ipynb`.

## Repository Contributors
* Tuan Nguyen Doan (Main developer) – [LinkedIn](https://www.linkedin.com/in/tuan-nguyen-doan) – tuandoan.nguyen@yale.edu
* Killian McLoughlin - [LinkedIn](www.linkedin.com/in/killian-mc-loughlin-5a151032) - killian.mcloughlin@yale.edu 
* William Brady - [Website](http://williamjbrady.com) – bradywilliamj@gmail.com 

## Citation
Brady, W.J., McLoughlin, K.L., Doan, T.N., & Crockett, M.J. (2019). DOC: Digital outrage classifier. Retrieved from www.github.com

## License
Distributed under the GNU General Public License v3.0 license. See ``LICENSE`` for more information.

## Release History
* 0.1.0
    * Internal release
    * UPDATE: created `README.md` file
    * UPDATE: updated from `MIT` licence to `General Public License version 3`
* 0.1.1
    * UPDATE: new model files uploaded
    * UPDATE: model re-training scripts pushed
* 0.1.2
    * UPDATE: updated `README.md` file
    * UPDATE: removed files from older models
    * UPDATE: removed old code blocks and comments
    * ADDED: link to supplementary materials
    * ADDED: contributors section
* 0.1.3
    * ADDED: citation section
    * ADDED: prototype citation
    * ADDED: link to OSF for project
    * UPDATED: `README.md` layout
* 0.1.4
    * UPDATED: package name to *DOC: Digital Outrage Classifier*
* 0.1.5
    * UPDATED: modularize the codebase of the project
* (TBD version)
    * UPDATED: refactor and modularize the codebase
    * UPDATED: `README.md` information
    * UPDATED: upload separatd `LICENSE` file
    * UPDATED: simplify `setup.py`, add installation requirement

<!-- Markdown link & img dfn's -->
[made-with-python]: https://img.shields.io/badge/Made%20with-Python-FF0000.svg
[outrage-image]: https://img.shields.io/badge/DOC-v0.1.4-orange.svg

[travis-image]: https://img.shields.io/travis/dbader/node-datadog-metrics/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/dbader/node-datadog-metrics
