#!bin/bash
conda create -n prism python 3.10 --yes
conda activate prism
python -m pip install --upgrade -r workstation/requirements.txt
python m23_04_29_run1.py