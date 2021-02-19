 ![image](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/images.jfif)
# Team Martell                          
* Marcin Szleszynski
* Michal Podlaszuk
* Bence
* Sai Mohan Reddy Dalli

## Build Week II Project: Jane Street Kaggle Competition [https://www.kaggle.com/c/jane-street-market-prediction]

## Data Preparation Report


![Overview](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/overview.PNG)

### Variables

![Variables](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/merge.PNG)

### Interactions

Date vs Response:
![interactions](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/date_vs_resp.PNG)

Weight vs Response
![weight](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/weight_vs_resp.PNG)

### Missing Values

![missing](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/Missing%20values.PNG)

### Correlations

Pearson Correlation:

![pearson](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/pearson_corr.PNG)


Heat Map:

![heat](https://github.com/MichalPodlaszuk/Build_Week_2/blob/main/pngs/heatmap.PNG)

### Data Cleaning and Preprocessing

* Dropped highly correlated features (i,e >0.9).
* Dropped the resp1, resp2, resp3, resp4 features.
* Filling the NAN's with zero's (i,e 0).
* Normalising the dataset using Standard Scaler.

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
### Seperating Features (X) and Target (y)

```python
train = pd.read_csv('jane_st.csv')

X = pd.concat([remove_collinear_features(feats, 0.9), train['weight'], train['ts_id']], axis = 1)

y = pd.Series([1 if x > 0 else 0 for x in train['resp']])
```

### Splitting the data

* Using Sklearn libraries
* Training --> 80%
* Testing  --> 20%

### Fitting the models

* Linear Model
* Logistic Regression
* Random Forest
* XGB classifier
* KNN
* GridSearchCV 

### Best model

```python
XGBClassifier(booster='gbtree', n_estimators=200, learning_rate=0.01, max_depth=30, use_label_encoder=False)
```

