# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:38:44 2021

@author: Jerry
"""

#apriori(records,min_support=0.045,min_confidence=0.2,min_lift=2,min_length=2)
#min_support the frequency that the data should occur at least
#min_confident is the probability that one is chosen and the other is also chosen which is the thing we want to know
#min_lift means after the association rule the probability 
#min_length means at least how much attribute can be put together 
#records = [[put attnum+answer and dec inside],[second data],[third data],[...],[...]]
#filter the result to the one is dec
import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from efficient_apriori import apriori

data = pd.read_csv(r'C:\Users\Administrator\Desktop\aprioriD.csv',header = None)
record = []
for i in range(0,111):
    record.append([str(data.values[i,j]) for j in range(0,24)])

itemsets, rules = apriori(record, min_support=0.1, min_confidence=0.6)
rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)
"""association_rule = apriori(record,min_support = 0.1,min_confident = 0.2,min_lift = 2,min_length = 2)
association = list(association_rule)
print(association)"""