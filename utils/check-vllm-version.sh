#!/bin/bash

grep -n -i vllm "$(python -c 'import trl,os; print(os.path.dirname(trl.__file__))')/import_utils.py"
