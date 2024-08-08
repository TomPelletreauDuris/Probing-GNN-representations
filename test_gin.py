
DATASET = "FC"

MODEL3 = "GIN3"
MODEL = MODEL3

import pickle as pkl
import networkx as nx
#load the properties
with open("results/"+DATASET+"_"+MODEL+"_train_properties.pkl", "rb") as f:
    train_properties = pkl.load(f)

with open("results/"+DATASET+"_"+MODEL+"_test_properties.pkl", "rb") as f:
    test_properties = pkl.load(f)

#print the first 5 properties
print(len(train_properties))
print(train_properties[0:5])

