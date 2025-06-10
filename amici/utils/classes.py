import os, re, json
import pandas as pd
import numpy as np
import sys
from collections import defaultdict

# Ensure path to data is relative to the script location
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '../data')

INDUSTRIES = []
SECTORS = []
SECTORS_TO_INDUSTRIES = defaultdict(list)
INDUSTRIES_TO_SECTORS = {}

with open(os.path.join(data_dir, 'ftm_industries.txt'), 'r') as f:
    for line in f:
        s, i = line.strip().split('_')
        INDUSTRIES.append(i)
        SECTORS.append(s)
        SECTORS_TO_INDUSTRIES[s].append(i)
        INDUSTRIES_TO_SECTORS[i] = s

SECTORS = list(set(SECTORS))
INDUSTRIES = list(set(INDUSTRIES))