# Team Martell - Jane Street Kaggle Competition

## TO-DO

[] test what we got so far with a kaggle submission

[] highlight insightful data visualisations (push it to pngs folder here)

[] plan for time series treatment

[] try feature extracting libraries (boruta)

[] look in kaggle code submissions for whatever we may lack

## Billboard

### Removing highly correlated features
```python
def remove_collinear_features(x, threshold):
    '''
    Objective:
        Remove collinear features in a dataframe with a correlation coefficient
        greater than the threshold. Removing collinear features can help a model 
        to generalize and improves the interpretability of the model.

    Inputs: 
        x: features dataframe
        threshold: features with correlations greater than this value are removed

    Output: 
        dataframe that contains only the non-highly-collinear features
    '''

    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                #print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)
    print('Removed Columns {}'.format(drops))
    return x
```
### X and y

```python
train = pd.read_csv('jane_st.csv')

X = pd.concat([remove_collinear_features(feats, 0.9), train['weight'], train['ts_id']], axis = 1)

y = pd.Series([1 if x > 0 else 0 for x in train['resp']])
```

### Best model

```python
XGBClassifier(booster='gbtree', n_estimators=200, learning_rate=0.01, max_depth=30, use_label_encoder=False)
```

