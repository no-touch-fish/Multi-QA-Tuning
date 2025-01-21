from functools import lru_cache
import os
import json
from glob import glob
import random
import re
from tqdm import tqdm
from itertools import groupby
import numpy as np
import requests
from transformers import pipeline
import torch

def reduce_CoT(cot):
    output = []
    while '**' in cot:
        index_1 = cot.find('**')
        index_2 = cot.find('\n')
        output.append(cot[index_1+2:index_2])
        cot = cot[index_2+1:]
    output.append(" " + cot)
    return ("").join(output)

cot = "How much money does Betty have in the beginning? ** In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50.\nHow much money did Betty's grandparents give her? ** Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30.\nHow much more money does Betty need to buy the wallet? ** This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more.\n#### 5"
cot = reduce_CoT(cot)
print(cot)





