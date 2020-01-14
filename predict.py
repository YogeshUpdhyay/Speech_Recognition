#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:42:14 2020

@author: Yogesh Upadhyay
"""

import pickle as pk
#import matplotlib.pyplot as plt
def predict(input_path):
    pickle_in = open('model.pickle','rb')
    model = pk.load(pickle_in)
    y = model.predict(input_path)
    return y
