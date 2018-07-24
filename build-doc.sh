#!/usr/bin/env bash

# script to build doc on appropriate format for github pages
# change build_dir in make file to none

source venv/bin/activate
cd docs
rm -rf *.html *.js *.inv _modules/ _static _sources/
sphinx-apidoc -e -d 3 -o source ../preprocessing/ tests/ __init__.py
make clean
make html
mv html/* .
