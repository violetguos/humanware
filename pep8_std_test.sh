#!/bin/bash

# this is a more primitive command for all .py files
# find . -name '*.py' -exec pycodestyle {} \; > pep8_report.txt

# We exclude eval.py and any legacy code in our evaluation of flake8.
# The following will give you all unique error codes found.
flake8 --ignore E501,E402,W,F --format='%(code)s' --exclude eval.py | sort | uniq > pep8_report1.txt