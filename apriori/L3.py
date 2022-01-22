import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
from efficient_apriori import apriori

data = pd.read_csv(r'C:\Users\Administrator\Desktop\apriori\L3.csv',header = None)
record = []
for i in range(0,22):
    record.append([str(data.values[i,j]) for j in range(0,24)])

itemsets, rules = apriori(record, min_support=0.05, min_confidence=0.8)
rules_rhs = filter(lambda rule: len(rule.lhs) == 2 and len(rule.rhs) == 1, rules)
for rule in sorted(rules_rhs, key=lambda rule: rule.lift):
    print(rule)
