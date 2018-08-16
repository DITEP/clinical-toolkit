clinical-toolkit
================

processing-toolbox is a machine learning python API designed for clinical data
processing.
The objective of this repository to gather in a same API useful tools for such
tasks rather than implementing new specific algorithms. That is the reason why
it relies on many widely used APIs such as `scikit-learn <http://scikit-learn.org>`_
or `gensim <https://radimrehurek.com/gensim/index.html>`_ .


Installation
------------
The package is not available on PyPI so you need to install it from source.

.. code-block:: sh
    :linenos:

    $ git clone https://github.com/DITEP/clinical-toolkit.git
    $ cd preprocessing-toolbox
    $ pip install -r requirements.txt
    $ pip install .


To check consistency of the installation, go to the root of the project and
run

.. code-block:: sh
    :linenos:

    $ nosetests

Which whould prompt `OK` in your terminal;


Dependencies
------------
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


Development
-----------
Everyone is welcome to take part to the project, there are however a few guidelines
one should need to follow to keep it clean and functioning

* make sure your code follows `PEP 8 <https://www.python.org/dev/peps/pep-0008>`_ guidelines
  for optimal readability. The best way to check it is by using `flake 8 <http://flake8.pycqa.org/en/latest/>`_
  tool.
* to report an issue or a bug, refer https://github.com/DITEP/clinical-toolkit/issues
  and make sur the error you report is **reproducible**.
* to make add a feature or fix a bug, clone the repo and create a new branch
  named *new_feature* and then make the pull request so that other contributors
  can review you code.



References
----------




