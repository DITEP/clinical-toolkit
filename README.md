# Processing Toolbox 

Full documentation available at https://ditep.github.io/processing-toolbox/

This repository contains scripts to preprocess a clinical database. It notably
includes functions to process medical text reports, and feature selection on 
high cardinality categorical features (eg: for medication names).

Many of the scripts rely on scikit-learn API


Disclaimer: This repository has not yet been tested for different databases,
and might not be fully compatible for many use cases. However, everyone is
welcome to start pull-requests and issues to improve the development of 
this <i>toolbox</i>.



## Installation
The package is not available on PyPI so you need to install it from source.

```bash
$ git clone https://github.com/DITEP/processing-toolbox.git
$ cd preprocessing-toolbox
$ pip install -r requirements.txt
$ pip install . 
```
You should then run the tests to check the consistency of the installation (see subsection <b> Testing </b>)



### Dependencies
The repository is compatible with following versions of packages but may also
work with previous versions. 

Python 2 has not been tested. Windows support has not yet been tested

* beautifulsoup4==4.6.0
* gensim==3.4.0
* nltk==3.3
* nose==1.3.7
* numpy==1.14.2
* pandas==0.23.0
* requests==2.18.4
* scikit-learn==0.19.1
* scipy==1.1.0
* SQLAlchemy==1.2.7
* Unidecode==1.0.22


###  Testing
Unit testing is performed using [nose](http://nose.readthedocs.io/en/latest/)
library which is both efficient and easy to use. 
However good, Nose is now in maintenance mode and the migration to another testing framework will probably be necessary 
in later development.

To launch the tests of a particular module:
```bash
$ cd path/to/module
$ nosetests tests # tests is directory that contains the tests scripts

........
----------------------------------------------------------------------
Ran 8 tests in 0.309s

OK
```
You can also perform all the tests at once by placing at the root directory
```bash
$ nosetests

..................
----------------------------------------------------------------------
Ran 19 tests in 16.747s

OK

```



## References



## TODO
* example notebooks
* contributing guidelines
* issue template

