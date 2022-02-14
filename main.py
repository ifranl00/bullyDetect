import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import matplotlib.pyplot as plt
import re


print("Hola mundo")

Data = pd.read_csv('DataSet/labeled_data.csv')

print(Data)
print(Data.head())
print(Data.shape)
print(Data.describe)
print(Data.dtypes)

for x in Data.columns:
    print(f"{x}\n{Data[x].unique()[:10]}")

print(Data['tweet'])

# get the number of missing data points per column
missing_values_count =Data.isnull().sum()

# look at the # of missing points in the first ten columns
print(missing_values_count[0:10])

# Preprocessing and data cleaning phase

def removeStrangeSymbols(column):
    #column = re.sub('@[^\s]+', '', column)
    #column = re.sub('RT', '', column)
    newStr =[]
    for x in column:
        newStr.append(x.replace("RT", ''))

    print(newStr)
    #column = ' '.join(word for word in column.split(' ') if not word.startswith('http')) #remove unuseful spaces
    return column

Data['tweet'] = Data['tweet'].astype(str)
x_array = pd.Series(Data['tweet'])
x_array = x_array.astype(str)
x_array = removeStrangeSymbols(x_array)

print(x_array)




print(Data['tweet'])