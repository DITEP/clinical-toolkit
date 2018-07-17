Quickstart
==========

This page introduces the basic concepts related to data analysis and processing
for clinical databases. The API has been originally developed for various
machine learning tasks in a context of early drug development for cancer, but it
is intended to be flexible and adaptable for other similar problematics.

The objective of the API is to make available the tools needed to solve the issues
raised by the complexity of Electronic Health Records (EHR), that will be
introduced in intro_ .



Introduction
------------
.. _intro:
    During the last decade, there has been tremendous progress in the field of
    of machine learning which has been conducted by the availability of massive
    amounts of data and the thriving power of computers. Those progress have
    impacted domains such as computer vision, speech recognition and many others.
    More recently, researchers have started to apply this knowledge on healthcare,
    given clinical data from hospitals or other portable devices and that is
    what we are focusing this API on.

    The usage of EHR has been widely adopted around the earth, which has led
    doctors and statisticians to mine them to improve the care of patients.
    However, medical health records are very difficult to tackle since they contain
    all the difficulties that exist in data analysis:
    * **sparcity**
    * **high cardinality categorical features**
    * **unstructured data** (text, images)
    * **temporality of the events**

    All those issues make it hard to initiate a machine learning project using
    clinical data, for those reason, we aim at providing the right tools to
    preprocess such databases with *efficiency* and *simplicity*.



