# machine-learning-pandas
how can you use pandas in python

# Pandas
#### pandas is a Python package providing fast, flexible, and expressive data structures designed
#### to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the
#### fundamental high-level building block for doing practical, real-world data analysis in Python.

# Data Structure
#### Pandas docs: In 0.20.0, Panel is deprecated and will be removed in a future version. The 3-
#### D structure of a Panel is much less common for many types of data analysis than the 1-D of
#### the Series or the 2-D of the DataFrame.

## 1 - Series

```python
import pandas as pd
my_series = pd.Series([1, 2, 3,4,5],index=['row1','row2','row3','row4','row5'])
# result
# row1 1
# row2 2
# row3 3
# row4 4
# row5 5
```

## Show Values
```python
my_series.values
# array([1, 2, 3, 4, 5], dtype=int64)
```

## Show index
```python
my_series.index
# Index(['row1', 'row2', 'row3', 'row4', 'row5'], dtype='object')
```

##  Set alphabet label as new index
```python
my_series.index = ['A','B','C','D','E']
```

# DataFrame
#### Pandas docs : Two-dimensional size-mutable, potentially heterogeneous tabular data
#### structure with labeled axes (rows and columns). Arithmetic operations align on both row and
#### column labels. Can be thought of as a dict-like container for Series objects. The primary
#### pandas data structure

## Create DataFrame with Array
```python
import numpy as np
my_array = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
my_df = pd.DataFrame(my_array, index=['row1' ,'row2' ,'row3' ,'row4'], columns=['col1' ,'col2' ,'col3' ,'col4'])
```

## Create DataFrame with Dictionary
```python
import numpy as np
my_array = np.array([[1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15], [4, 8, 12, 16]])
my_df = pd.DataFrame(my_array, index=['row1' ,'row2' ,'row3' ,'row4'], columns=['col1' ,'col2' ,'col3' ,'col4'])
```

## Selecting 1
```python
array.loc['row1']['col2']
```
---------------------
## Selecting 2
```python
array.iloc[2][3]
```

## Edit a DataFrame
```python
array.['col5'] = [20 ,21 ,22 ,23]
# or
array.loc['row1', 'col1'] = 0
```

## Reset index
```python
array.reset_index(drop=True)
```

## Deleting
```python
array.drop('col5',axis=1)
# or
array.drop('row1',axis=0)
```

## Renaming
```python
array.rename(columns={'col4':'col_four'})
```

## sort values
```python
array.sort_values(by='col1',ascending=False)
# or 
array.sort_values(by='col1',ascending=True)
```

## head !!!
it can show you five first rows
```python
array.head()
```

I hope this data will be useful to you.






