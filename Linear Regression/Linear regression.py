# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 16:32:02 2018

@author: Rohit
"""

"""https://archive.ics.uci.edu/ml/datasets/Air+Quality"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
x=np.array(data[0:5,:])
theta=np.zeros((len(x),1))

