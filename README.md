# Titanic: Machine Learning from Disaster

# The Challenge

The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered “unsinkable” RMS Titanic sank after colliding with an iceberg. Unfortunately, there weren’t enough lifeboats for everyone onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers the question: “what sorts of people were more likely to survive?” using passenger data (ie name, age, gender, socio-economic class, etc). 

# Tools we would need


```python
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, median_absolute_error, max_error, explained_variance_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

import xgboost as xgb
from xgboost import XGBRegressor, plot_importance 

import lightgbm as lgb

from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from keras.optimizers import RMSprop 
from keras.callbacks import EarlyStopping 
import keras

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import threading
import re

sns.set()

TrainData =  pd.read_csv('input/train.csv')
TestData = pd.read_csv('input/test.csv')
```

    Using TensorFlow backend.


# Feature engineering

## Interpolation of missing values

One can notice that in traing and testing data sets combined there are:
- **1014** passengers with missing `Cabin`
- **418** passengers with missing `Survived` value
- **263** passengers with missing `Age`
- **2** passengers with missing `Embarked` port
- **1** record with missing `Fare` value

We will try to approximate those values below.


```python
traind = TrainData.copy()
testd = TestData.copy()
td_merged = traind.append(testd, ignore_index=True)
td_merged.info() 
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 12 columns):
     #   Column       Non-Null Count  Dtype  
    ---  ------       --------------  -----  
     0   PassengerId  1309 non-null   int64  
     1   Survived     891 non-null    float64
     2   Pclass       1309 non-null   int64  
     3   Name         1309 non-null   object 
     4   Sex          1309 non-null   object 
     5   Age          1046 non-null   float64
     6   SibSp        1309 non-null   int64  
     7   Parch        1309 non-null   int64  
     8   Ticket       1309 non-null   object 
     9   Fare         1308 non-null   float64
     10  Cabin        295 non-null    object 
     11  Embarked     1307 non-null   object 
    dtypes: float64(3), int64(4), object(5)
    memory usage: 122.8+ KB


### Retrieve Title from Name


```python
# Retrieve title from Name, we will use it as a feature
for index, person in td_merged.iterrows():
    found = re.match(r".+ (.+\.).*", person.Name)
    if found:
        td_merged.loc[index, 'Title'] = found.group(1)

td_filtered = td_merged.drop(['PassengerId','Name'],axis=1)
```

### Fill missing `Embarked` values


```python
# According to publicly available report, those 2 ladies embarked in Southampton
td_filtered[['Embarked']] = td_filtered[['Embarked']].fillna('S')
```

### Fill missing `Fare` value


```python
# There's only 1 missing fare for a senior citezen, we'll use a median fare
# among other people on board of the same age
missing_fare = float(td_filtered[['Pclass','Age','Fare']]
                     [(td_filtered.Age > 50) & (td_filtered.Pclass == 3)]
                     .groupby(['Pclass']).median().Fare)
td_filtered[['Fare']] = td_filtered[['Fare']].fillna(missing_fare)
```

### Use `Cabin`'s first letter as sector identifier on the ship


```python
# Use first cabin latter to distinguish location on the ship
td_filtered.Cabin = td_filtered.Cabin[td_filtered.Cabin.notnull()].apply(lambda c: c[0])
```

## Categorical feature encoding
* `Sex` - we have discovered only 2 possible genders in the data: _male_ and _female_
* `Ticket` - some tickets are reoccurring, perhaps shared among passengers tagging along in group (family members, tourists, friends, etc.)
* `Cabin` - Some cabins are also shared among passengers
* `Embarked` - this feature has 3 ports, some of the passengers (2 ladies from _Southampton_ ) have this field missing, we will manually populate those entries, as this information is publically available
* `Title` - We will extract title (_Mr._ , _Ms._ , _Mrs._ , etc.) from the name. Perhaps, it's worth to have a look at surname, and play around with that feature to predict survival rate of particular family members but for now we will drop it and use title only.

Apart from `Name`, we are also dropping `PassengerId` to avoid our models allocating excessive weights to irrelevant features.


```python
categorical_features = ['Sex','Cabin','Embarked', 'Ticket', 'Title']
encoders = dict()
for feature in categorical_features:
    enc = LabelEncoder()
    encoders[feature] = enc
    td_filtered.loc[:,feature] = enc.fit_transform(td_filtered[feature].astype(str))
```

## Feature relationship analysis

In this step we will discover how different features correlate with each other and prepare a set of important variables to feed to our predictive models later.

### Heatmap


```python
pd.set_option('precision',2)
plt.figure(figsize=(10, 8))
sns.heatmap(td_filtered.corr())
plt.suptitle("Pearson Correlation Heatmap")
plt.show();
td_filtered.corr()
```


![png](output_18_0.png)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>Title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Survived</th>
      <td>1.00</td>
      <td>-0.34</td>
      <td>-0.54</td>
      <td>-0.08</td>
      <td>-3.53e-02</td>
      <td>0.08</td>
      <td>-0.17</td>
      <td>0.26</td>
      <td>-3.01e-01</td>
      <td>-0.17</td>
      <td>-0.20</td>
    </tr>
    <tr>
      <th>Pclass</th>
      <td>-0.34</td>
      <td>1.00</td>
      <td>0.12</td>
      <td>-0.41</td>
      <td>6.08e-02</td>
      <td>0.02</td>
      <td>0.31</td>
      <td>-0.56</td>
      <td>7.35e-01</td>
      <td>0.19</td>
      <td>0.02</td>
    </tr>
    <tr>
      <th>Sex</th>
      <td>-0.54</td>
      <td>0.12</td>
      <td>1.00</td>
      <td>0.06</td>
      <td>-1.10e-01</td>
      <td>-0.21</td>
      <td>0.02</td>
      <td>-0.19</td>
      <td>1.25e-01</td>
      <td>0.10</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>Age</th>
      <td>-0.08</td>
      <td>-0.41</td>
      <td>0.06</td>
      <td>1.00</td>
      <td>-2.44e-01</td>
      <td>-0.15</td>
      <td>-0.09</td>
      <td>0.18</td>
      <td>-3.12e-01</td>
      <td>-0.08</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>SibSp</th>
      <td>-0.04</td>
      <td>0.06</td>
      <td>-0.11</td>
      <td>-0.24</td>
      <td>1.00e+00</td>
      <td>0.37</td>
      <td>0.06</td>
      <td>0.16</td>
      <td>7.95e-03</td>
      <td>0.07</td>
      <td>-0.18</td>
    </tr>
    <tr>
      <th>Parch</th>
      <td>0.08</td>
      <td>0.02</td>
      <td>-0.21</td>
      <td>-0.15</td>
      <td>3.74e-01</td>
      <td>1.00</td>
      <td>0.05</td>
      <td>0.22</td>
      <td>-3.44e-02</td>
      <td>0.04</td>
      <td>-0.09</td>
    </tr>
    <tr>
      <th>Ticket</th>
      <td>-0.17</td>
      <td>0.31</td>
      <td>0.02</td>
      <td>-0.09</td>
      <td>6.39e-02</td>
      <td>0.05</td>
      <td>1.00</td>
      <td>-0.01</td>
      <td>2.33e-01</td>
      <td>0.03</td>
      <td>-0.01</td>
    </tr>
    <tr>
      <th>Fare</th>
      <td>0.26</td>
      <td>-0.56</td>
      <td>-0.19</td>
      <td>0.18</td>
      <td>1.60e-01</td>
      <td>0.22</td>
      <td>-0.01</td>
      <td>1.00</td>
      <td>-5.47e-01</td>
      <td>-0.24</td>
      <td>-0.08</td>
    </tr>
    <tr>
      <th>Cabin</th>
      <td>-0.30</td>
      <td>0.73</td>
      <td>0.13</td>
      <td>-0.31</td>
      <td>7.95e-03</td>
      <td>-0.03</td>
      <td>0.23</td>
      <td>-0.55</td>
      <td>1.00e+00</td>
      <td>0.23</td>
      <td>0.05</td>
    </tr>
    <tr>
      <th>Embarked</th>
      <td>-0.17</td>
      <td>0.19</td>
      <td>0.10</td>
      <td>-0.08</td>
      <td>6.56e-02</td>
      <td>0.04</td>
      <td>0.03</td>
      <td>-0.24</td>
      <td>2.31e-01</td>
      <td>1.00</td>
      <td>0.06</td>
    </tr>
    <tr>
      <th>Title</th>
      <td>-0.20</td>
      <td>0.02</td>
      <td>0.22</td>
      <td>0.27</td>
      <td>-1.76e-01</td>
      <td>-0.09</td>
      <td>-0.01</td>
      <td>-0.08</td>
      <td>4.63e-02</td>
      <td>0.06</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
</div>



### Closer look on `Gender`, `Age` and `Class` 

From age and gender insights we can observe that passengers of the First class got the best survival rate followed by the Second class and the Third as the last one. Probably because passengers of the 3rd class ended up to be locked in their cabins, that might have been done to reduce sinking rate of the ship so that more lifes could have be saved.


```python
sns.catplot(x='Sex', y='Age', hue='Survived', data=td_filtered, kind="box", col='Pclass',
               palette=['lightpink','lightgreen'])
```




    <seaborn.axisgrid.FacetGrid at 0x7f5924745b00>




![png](output_21_1.png)


## Feature recovery

### Regression analysis of `Age`

For this model we are going to drop `Ticket` feature here, as it reduces error rate of prediction.


```python
td_with_age = td_filtered[(~td_filtered.Age.isnull())]
td_without_age = td_filtered[(td_filtered.Age.isnull())].drop(['Age'], axis=1)

X = td_with_age.drop(['Age'], axis=1)
Y = td_with_age.Age

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=10)

```

#### Gradient boosting with LightGBM


```python
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq': 5,
    'num_threads': threading.active_count(),
}

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

modelLGB = lgb.train(params,
                lgb_train,
                num_boost_round=50,
                valid_sets=lgb_eval,
                early_stopping_rounds=10, verbose_eval=0)

predictions = modelLGB.predict(X_test, num_iteration=modelLGB.best_iteration)

print('R2 Score (best is 1.0): %s' % r2_score(y_test.to_numpy(), predictions))
print('MedAE (the smaller the better): %s' % median_absolute_error(y_true=y_test.to_numpy(),y_pred=predictions))
print('Max Error: %s' % max_error(y_true=y_test.to_numpy(),y_pred=predictions))
print('ExpVar score (best is 1.0): %s' % explained_variance_score(y_true=y_test.to_numpy(),y_pred=predictions))
print('RMSE:', mean_squared_error(y_test, predictions) ** 0.5)

feature_importance = pd.DataFrame()
feature_importance['Score'] = modelLGB.feature_importance()
feature_importance['Feature'] = modelLGB.feature_name()
feature_importance = feature_importance.sort_values(by='Score',ascending=False)

ax = sns.barplot(x='Score',y='Feature',data=feature_importance, palette="Purples_d", orient='h')
```

    R2 Score (best is 1.0): 0.410205426474857
    MedAE (the smaller the better): 7.275197240862298
    Max Error: 33.729644507969894
    ExpVar score (best is 1.0): 0.41303551808169436
    RMSE: 11.593578558801603



![png](output_26_1.png)


#### Gradient boosting with XGBoost


```python
modelXGB = XGBRegressor(n_estimators=10000)
modelXGB.fit(X_train, y_train, early_stopping_rounds=10, 
             eval_set=[(X_test, y_test)], verbose=False)
predictions = modelXGB.predict(X_test)
print('R2 Score (best is 1.0): %s' % r2_score(y_test.to_numpy(), predictions))
print('MedAE (the smaller the better): %s' % median_absolute_error(y_true=y_test.to_numpy(),y_pred=predictions))
print('Max Error: %s' % max_error(y_true=y_test.to_numpy(),y_pred=predictions))
print('ExpVar score (best is 1.0): %s' % explained_variance_score(y_true=y_test.to_numpy(),y_pred=predictions))
print('RMSE:', mean_squared_error(y_test, predictions) ** 0.5)

temp = pd.DataFrame()
temp['Actual'] = y_test
temp['Predicted'] = predictions
sns.regplot(x='Actual',y='Predicted',data=temp)

plot_importance(modelXGB)

plt.show()
```

    R2 Score (best is 1.0): 0.3521010399536765
    MedAE (the smaller the better): 6.317258834838867
    Max Error: 36.57940483093262
    ExpVar score (best is 1.0): 0.3582935130372745
    RMSE: 12.15124463439559



![png](output_28_1.png)



![png](output_28_2.png)


We will go with the results produced by **XGBoost** in this case as the predictions indicates slightly lower error rates comparing to **LightGBM**

#### Populating missing `Age` values with results of our prediction


```python
predictions = modelXGB.predict(td_without_age)
td_age_restored = td_without_age.copy()
td_age_restored.loc[:,'Age'] = predictions
td_filtered.update(td_with_age.append(td_age_restored))
td_filtered.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 11 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    float64
     1   Pclass    1309 non-null   int64  
     2   Sex       1309 non-null   int64  
     3   Age       1309 non-null   float64
     4   SibSp     1309 non-null   int64  
     5   Parch     1309 non-null   int64  
     6   Ticket    1309 non-null   int64  
     7   Fare      1309 non-null   float64
     8   Cabin     1309 non-null   int64  
     9   Embarked  1309 non-null   int64  
     10  Title     1309 non-null   int64  
    dtypes: float64(3), int64(8)
    memory usage: 112.6 KB


### Classification of missing `Cabin` (sector of the ship) values


```python
td_with_cabin = td_filtered[(td_filtered.Cabin.notna())].drop(['Survived'], axis=1)
td_without_cabin = td_filtered[(td_filtered.Cabin.isna())].drop(['Cabin', 'Survived'], axis=1)

X = td_with_cabin.drop(['Cabin'], axis=1)

input_features = X.columns.values
n_input_features = len(input_features)
n_output_feature = len(td_with_cabin.Cabin.unique())

Y = td_with_cabin.Cabin

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=10)
```

#### Predicting with multiple layer Neural Networks (Keras & TF)


```python
modelNN = Sequential([
    Dense(n_input_features, input_dim=n_input_features, activation='relu'),
    Dense(n_input_features * 8, activation='relu'),
    Dense(n_input_features * 8, activation='relu'),
    Dropout(0.1),
    Dense(n_output_feature, activation='softmax'),
])
modelNN.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
modelNN.summary()
modelNN.fit(
    X_train, y_train,
    epochs=30,
    validation_data=(X_test, y_test),
    verbose = 1
)
loss, accuracy = modelNN.evaluate(X_test, y_test)
print("Accuracy: %s" % accuracy)
print("Loss: %s" % loss)
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 9)                 90        
    _________________________________________________________________
    dense_2 (Dense)              (None, 72)                720       
    _________________________________________________________________
    dense_3 (Dense)              (None, 72)                5256      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 72)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 9)                 657       
    =================================================================
    Total params: 6,723
    Trainable params: 6,723
    Non-trainable params: 0
    _________________________________________________________________
    Train on 1243 samples, validate on 66 samples
    Epoch 1/30
    1243/1243 [==============================] - 0s 184us/step - loss: 20.5537 - accuracy: 0.5430 - val_loss: 2.4356 - val_accuracy: 0.7727
    Epoch 2/30
    1243/1243 [==============================] - 0s 49us/step - loss: 5.4559 - accuracy: 0.6959 - val_loss: 1.1069 - val_accuracy: 0.8182
    Epoch 3/30
    1243/1243 [==============================] - 0s 57us/step - loss: 3.2069 - accuracy: 0.6822 - val_loss: 0.7519 - val_accuracy: 0.8485
    Epoch 4/30
    1243/1243 [==============================] - 0s 48us/step - loss: 2.1670 - accuracy: 0.7120 - val_loss: 0.4772 - val_accuracy: 0.8485
    Epoch 5/30
    1243/1243 [==============================] - 0s 60us/step - loss: 1.4175 - accuracy: 0.7570 - val_loss: 0.5681 - val_accuracy: 0.8030
    Epoch 6/30
    1243/1243 [==============================] - 0s 48us/step - loss: 1.1371 - accuracy: 0.7643 - val_loss: 0.4783 - val_accuracy: 0.8030
    Epoch 7/30
    1243/1243 [==============================] - 0s 53us/step - loss: 1.0123 - accuracy: 0.7755 - val_loss: 0.5597 - val_accuracy: 0.8030
    Epoch 8/30
    1243/1243 [==============================] - 0s 61us/step - loss: 0.9108 - accuracy: 0.7860 - val_loss: 0.5117 - val_accuracy: 0.8333
    Epoch 9/30
    1243/1243 [==============================] - 0s 56us/step - loss: 0.8819 - accuracy: 0.7916 - val_loss: 0.4689 - val_accuracy: 0.8636
    Epoch 10/30
    1243/1243 [==============================] - 0s 51us/step - loss: 0.8671 - accuracy: 0.7940 - val_loss: 0.5196 - val_accuracy: 0.8485
    Epoch 11/30
    1243/1243 [==============================] - 0s 54us/step - loss: 0.7799 - accuracy: 0.7916 - val_loss: 0.5808 - val_accuracy: 0.8182
    Epoch 12/30
    1243/1243 [==============================] - 0s 57us/step - loss: 0.7371 - accuracy: 0.8061 - val_loss: 0.4785 - val_accuracy: 0.8485
    Epoch 13/30
    1243/1243 [==============================] - 0s 49us/step - loss: 0.7820 - accuracy: 0.7932 - val_loss: 0.4530 - val_accuracy: 0.8485
    Epoch 14/30
    1243/1243 [==============================] - 0s 48us/step - loss: 0.7369 - accuracy: 0.7965 - val_loss: 0.5625 - val_accuracy: 0.8030
    Epoch 15/30
    1243/1243 [==============================] - 0s 56us/step - loss: 0.6974 - accuracy: 0.7997 - val_loss: 0.5419 - val_accuracy: 0.8182
    Epoch 16/30
    1243/1243 [==============================] - 0s 50us/step - loss: 0.7342 - accuracy: 0.7949 - val_loss: 0.4874 - val_accuracy: 0.8939
    Epoch 17/30
    1243/1243 [==============================] - 0s 56us/step - loss: 0.7067 - accuracy: 0.7989 - val_loss: 0.5525 - val_accuracy: 0.8333
    Epoch 18/30
    1243/1243 [==============================] - 0s 48us/step - loss: 0.6865 - accuracy: 0.8117 - val_loss: 0.4650 - val_accuracy: 0.8636
    Epoch 19/30
    1243/1243 [==============================] - 0s 55us/step - loss: 0.6357 - accuracy: 0.8109 - val_loss: 0.7712 - val_accuracy: 0.8030
    Epoch 20/30
    1243/1243 [==============================] - 0s 57us/step - loss: 0.7217 - accuracy: 0.8077 - val_loss: 0.4486 - val_accuracy: 0.8485
    Epoch 21/30
    1243/1243 [==============================] - 0s 48us/step - loss: 0.6407 - accuracy: 0.8142 - val_loss: 0.5779 - val_accuracy: 0.8030
    Epoch 22/30
    1243/1243 [==============================] - 0s 55us/step - loss: 0.6781 - accuracy: 0.8142 - val_loss: 0.6176 - val_accuracy: 0.7879
    Epoch 23/30
    1243/1243 [==============================] - 0s 49us/step - loss: 0.6365 - accuracy: 0.8061 - val_loss: 0.4864 - val_accuracy: 0.8333
    Epoch 24/30
    1243/1243 [==============================] - 0s 58us/step - loss: 0.6261 - accuracy: 0.8093 - val_loss: 0.5414 - val_accuracy: 0.8485
    Epoch 25/30
    1243/1243 [==============================] - 0s 49us/step - loss: 0.6280 - accuracy: 0.8077 - val_loss: 0.5224 - val_accuracy: 0.8485
    Epoch 26/30
    1243/1243 [==============================] - 0s 57us/step - loss: 0.6260 - accuracy: 0.8061 - val_loss: 0.5400 - val_accuracy: 0.8333
    Epoch 27/30
    1243/1243 [==============================] - 0s 48us/step - loss: 0.6215 - accuracy: 0.8077 - val_loss: 0.5785 - val_accuracy: 0.8182
    Epoch 28/30
    1243/1243 [==============================] - 0s 54us/step - loss: 0.6198 - accuracy: 0.8190 - val_loss: 0.4434 - val_accuracy: 0.8636
    Epoch 29/30
    1243/1243 [==============================] - 0s 53us/step - loss: 0.6120 - accuracy: 0.8069 - val_loss: 0.4659 - val_accuracy: 0.8485
    Epoch 30/30
    1243/1243 [==============================] - 0s 50us/step - loss: 0.6192 - accuracy: 0.8158 - val_loss: 0.5983 - val_accuracy: 0.8485
    66/66 [==============================] - 0s 49us/step
    Accuracy: 0.8484848737716675
    Loss: 0.5982763862068002


##### Model overview

The result produced by the model: **74% of accuracy** and **65% of loss** is somewhat not satisfying at all for this task!

#### Trying to classify with XGBoost


```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = { 
    'objective': 'multi:softmax',
    'learning_rate': 0.3,
    'max_depth': 6,
    'eta': 0.1,
    'nthread': threading.active_count(),
    'num_class': n_output_feature,
}
modelXGB = xgb.train(params=params,dtrain=dtrain,num_boost_round=100)
predictions = modelXGB.predict(dtest)
```

##### Model overview

Excellent result - 95% of accuracy!


```python
print('Accuracy: %s' % accuracy_score(y_test, predictions))
print('Error rate: %s' % (np.sum(predictions != y_test) / y_test.shape[0]))
print('MSE: %s' % mean_squared_error(y_test, predictions))
print('MedAE: %s' % median_absolute_error(y_true=y_test,y_pred=predictions))
print('Max Error: %s' % max_error(y_true=y_test,y_pred=predictions))
print('ExpVar score: %s' % explained_variance_score(y_true=y_test,y_pred=predictions))
```

    Accuracy: 0.9545454545454546
    Error rate: 0.045454545454545456
    MSE: 1.3333333333333333
    MedAE: 0.0
    Max Error: 6.0
    ExpVar score: 0.7780525502318393


#### Populating predicted values


```python
predictions = modelXGB.predict(xgb.DMatrix(td_without_cabin))
missing_cabins = np.array(predictions,dtype=int)
td_cabins_restored = td_without_cabin.copy()
td_cabins_restored.loc[:,'Cabin'] = missing_cabins
td_filtered.update(td_with_cabin.append(td_cabins_restored))
td_filtered['Survived'] = TrainData.Survived
td_filtered.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1309 entries, 0 to 1308
    Data columns (total 11 columns):
     #   Column    Non-Null Count  Dtype  
    ---  ------    --------------  -----  
     0   Survived  891 non-null    float64
     1   Pclass    1309 non-null   int64  
     2   Sex       1309 non-null   int64  
     3   Age       1309 non-null   float64
     4   SibSp     1309 non-null   int64  
     5   Parch     1309 non-null   int64  
     6   Ticket    1309 non-null   int64  
     7   Fare      1309 non-null   float64
     8   Cabin     1309 non-null   int64  
     9   Embarked  1309 non-null   int64  
     10  Title     1309 non-null   int64  
    dtypes: float64(3), int64(8)
    memory usage: 112.6 KB


Now we have all of our missing values recovered except the only one `Survived`. 

Let's jump to our final goal -- guess who survived and who did not.

### Classification of `Survived` passengers


```python
training_data = td_filtered[(~td_filtered.Survived.isnull())].drop(['Embarked'],axis=1)
testing_data = td_filtered[(td_filtered.Survived.isnull())].drop(['Embarked','Survived'],axis=1)

X = training_data.drop(['Survived'],axis=1)
Y = training_data.Survived

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.05, random_state=10)
```

#### Binary logistic regression with `LightGBM`


```python
param = {
    'objective': 'binary',
    'learning_rate': 0.05,
    'feature_fraction': 1,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'num_threads': threading.active_count(),
}
param['metric'] = ['auc', 'binary_logloss']

# create dataset for lightgbm
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

bst = lgb.train(param, lgb_train, num_boost_round=50, valid_sets=lgb_eval, early_stopping_rounds=10)
```

    [1]	valid_0's auc: 0.887019	valid_0's binary_logloss: 0.601159
    Training until validation scores don't improve for 10 rounds
    [2]	valid_0's auc: 0.883413	valid_0's binary_logloss: 0.581927
    [3]	valid_0's auc: 0.883413	valid_0's binary_logloss: 0.564348
    [4]	valid_0's auc: 0.884615	valid_0's binary_logloss: 0.550044
    [5]	valid_0's auc: 0.884615	valid_0's binary_logloss: 0.536333
    [6]	valid_0's auc: 0.885817	valid_0's binary_logloss: 0.52349
    [7]	valid_0's auc: 0.882212	valid_0's binary_logloss: 0.510005
    [8]	valid_0's auc: 0.884615	valid_0's binary_logloss: 0.499404
    [9]	valid_0's auc: 0.887019	valid_0's binary_logloss: 0.48808
    [10]	valid_0's auc: 0.884615	valid_0's binary_logloss: 0.478776
    [11]	valid_0's auc: 0.899038	valid_0's binary_logloss: 0.467515
    [12]	valid_0's auc: 0.908654	valid_0's binary_logloss: 0.456432
    [13]	valid_0's auc: 0.911058	valid_0's binary_logloss: 0.446728
    [14]	valid_0's auc: 0.911058	valid_0's binary_logloss: 0.438962
    [15]	valid_0's auc: 0.911058	valid_0's binary_logloss: 0.431808
    [16]	valid_0's auc: 0.913462	valid_0's binary_logloss: 0.424851
    [17]	valid_0's auc: 0.901442	valid_0's binary_logloss: 0.420208
    [18]	valid_0's auc: 0.901442	valid_0's binary_logloss: 0.416105
    [19]	valid_0's auc: 0.899038	valid_0's binary_logloss: 0.412113
    [20]	valid_0's auc: 0.889423	valid_0's binary_logloss: 0.409081
    [21]	valid_0's auc: 0.889423	valid_0's binary_logloss: 0.40384
    [22]	valid_0's auc: 0.884615	valid_0's binary_logloss: 0.398509
    [23]	valid_0's auc: 0.887019	valid_0's binary_logloss: 0.394099
    [24]	valid_0's auc: 0.887019	valid_0's binary_logloss: 0.39074
    [25]	valid_0's auc: 0.889423	valid_0's binary_logloss: 0.388734
    [26]	valid_0's auc: 0.891827	valid_0's binary_logloss: 0.384603
    Early stopping, best iteration is:
    [16]	valid_0's auc: 0.913462	valid_0's binary_logloss: 0.424851


##### Model overview


```python
print('Accuracy: %s' % bst.best_score['valid_0']['auc'])
print('Binary logloss: %s' % bst.best_score['valid_0']['binary_logloss'])

feature_importance = pd.DataFrame()
feature_importance['Score'] = bst.feature_importance()
feature_importance['Feature'] = bst.feature_name()
feature_importance = feature_importance.sort_values(by='Score',ascending=False)

ax = sns.barplot(x='Score',y='Feature',data=feature_importance, palette="Purples_d", orient='h')
```

    Accuracy: 0.9134615384615384
    Binary logloss: 0.4248511598054666



![png](output_50_1.png)


#### Binary logistic regression with `XGBoost`

Despite the name of the algorithm suggests the word "regression" we're going to perform classification underneath.

In order to answer to the question like "Did this passenger survive the disaster?" -- the possible answer can contain only 2 possible outcomes: `True` or `False`.

Let's see what rate of accuracy we can achieve with this model.


```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)
params = { 
    'objective': 'binary:logistic',
    'learning_rate': 0.05,
    'max_depth': 6,
    'eta': 0.01,
    'nthread': threading.active_count(),
}
modelXGB = xgb.train(params=params,dtrain=dtrain,num_boost_round=40)
predictions = modelXGB.predict(dtest)
```

##### Model overview

We achieved quite a good result:



```python
print('Accuracy: %s' % accuracy_score(y_test, predictions.round()))
print('MSE: %s' % mean_squared_error(y_test, predictions.round()))
```

    Accuracy: 0.8888888888888888
    MSE: 0.1111111111111111


##### Decision tree


```python
xgb.to_graphviz(modelXGB)
```




![svg](output_56_0.svg)



##### Feature importance

- From my point of view it doesn't look so obvious why our model picked up `Ticket` as the most important feature. My guess here would be that passengers in group has the best chance to escape the sinking ship. But strange that eliminating this feature gives worse accuracy score. So we will keep it as it is.

- Secondly, looks like `Age` contributes heavily to survival probability after whether the passenger was in group or alone. This drives to conclusion that passengers of particular age group have higher chances to survive.

- `Fare` and `Cabin` sector are the next deciders in this game of dice.

- Class and gender don't play so crucial role as it might appeared in the beginning.

- Apperantly a particular `Title` or whether the passenger is a parent or children (`Parch`) contribute very little to importance


```python
plot_importance(modelXGB)
plt.show()
```


![png](output_58_0.png)


#### KNN - K Nearest Neighbours with `scikit-learn`


```python
modelKNN = KNeighborsClassifier(n_neighbors=10)
modelKNN.fit(X_train, y_train)
predictions = modelKNN.predict(X_test)
```

##### Model overview

Quite a surprise -- KNN and Gradient Boosting ended up predicting with the same rate of accuracy!


```python
print('Accuracy: %s' % accuracy_score(y_test, predictions.round()))
print('MSE: %s' % mean_squared_error(y_test, predictions.round()))
```

    Accuracy: 0.8222222222222222
    MSE: 0.17777777777777778


#### Neural network with multiple layers (`Keras` & `TF`)


```python
neurons=len(testing_data.columns.values)
modelNN = Sequential([
    Dense(neurons * 8, input_dim=neurons, activation='relu'),
    Dense(neurons * 3, activation='relu'),
    Dense(neurons * 1, activation='relu'),
    Dense(1, activation='sigmoid'),
])

modelNN.compile(
    loss='mse',
    optimizer='adam',
    metrics=['accuracy']
)

modelNN.fit(
    X_train, y_train,
    epochs=5, batch_size=10,
    validation_data=(X_test, y_test),
    verbose = 1
)
```

    Train on 846 samples, validate on 45 samples
    Epoch 1/5
    846/846 [==============================] - 0s 247us/step - loss: 0.3488 - accuracy: 0.6395 - val_loss: 0.2676 - val_accuracy: 0.7333
    Epoch 2/5
    846/846 [==============================] - 0s 120us/step - loss: 0.3361 - accuracy: 0.6430 - val_loss: 0.2851 - val_accuracy: 0.7111
    Epoch 3/5
    846/846 [==============================] - 0s 115us/step - loss: 0.3224 - accuracy: 0.6667 - val_loss: 0.2672 - val_accuracy: 0.7333
    Epoch 4/5
    846/846 [==============================] - 0s 137us/step - loss: 0.3125 - accuracy: 0.6655 - val_loss: 0.2895 - val_accuracy: 0.6889
    Epoch 5/5
    846/846 [==============================] - 0s 133us/step - loss: 0.3108 - accuracy: 0.6690 - val_loss: 0.2221 - val_accuracy: 0.7778





    <keras.callbacks.callbacks.History at 0x7f587c501f28>



##### Model overview


```python
loss, accuracy = modelNN.evaluate(X_test, y_test)
print("Accuracy: %s" % accuracy)
print("Loss: %s" % loss)
```

    45/45 [==============================] - 0s 72us/step
    Accuracy: 0.7777777910232544
    Loss: 0.22205772731039258


#### Convolutional Neural Network with 1x1 dimension (`Keras` & `TF`)


```python
modelCNN = Sequential([
    Conv1D(filters=32, kernel_size=4, input_shape=(neurons,1)),
    Conv1D(filters=32, kernel_size=4, activation='relu'),
    MaxPooling1D(pool_size=2),
    Dropout(0.1),
    Flatten(),
    Dense(neurons * 4, activation='relu'),
    Dense(1, activation='sigmoid'),
])

modelCNN.compile(
   loss = 'mse',
   optimizer = 'adam',
   metrics = ['accuracy']
)

X_train_reshaped = X_train.to_numpy().reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_reshaped = X_test.to_numpy().reshape(X_test.shape[0], X_test.shape[1], 1)

modelCNN.fit(
    X_train_reshaped, y_train,
    epochs=5, batch_size=10,
    validation_data=(X_test_reshaped, y_test),
    verbose = 1
)
```

    Train on 846 samples, validate on 45 samples
    Epoch 1/5
    846/846 [==============================] - 0s 350us/step - loss: 0.4001 - accuracy: 0.5969 - val_loss: 0.2889 - val_accuracy: 0.7111
    Epoch 2/5
    846/846 [==============================] - 0s 156us/step - loss: 0.3894 - accuracy: 0.6076 - val_loss: 0.3025 - val_accuracy: 0.6889
    Epoch 3/5
    846/846 [==============================] - 0s 153us/step - loss: 0.3306 - accuracy: 0.6501 - val_loss: 0.2845 - val_accuracy: 0.7111
    Epoch 4/5
    846/846 [==============================] - 0s 160us/step - loss: 0.5733 - accuracy: 0.4196 - val_loss: 0.7111 - val_accuracy: 0.2889
    Epoch 5/5
    846/846 [==============================] - 0s 144us/step - loss: 0.5994 - accuracy: 0.3995 - val_loss: 0.7108 - val_accuracy: 0.2889





    <keras.callbacks.callbacks.History at 0x7f587c249f60>



##### Model overview

Worst result! I know... this domain of problem is not suitable for CNN as the data set isn't large enough, so the model ended up underfitted -- I'm just experimenting :)


```python
loss, accuracy = modelCNN.evaluate(X_test_reshaped, y_test)
print("Accuracy: %s" % accuracy)
print("Loss: %s" % loss)
```

    45/45 [==============================] - 0s 83us/step
    Accuracy: 0.2888889014720917
    Loss: 0.7108210298750136


### Feed testing data set


```python
TestDataReshaped = testing_data.to_numpy().reshape(testing_data.shape[0],testing_data.shape[1],1)
OutputCNN = testing_data.copy()
OutputCNN["Survived"] = modelCNN.predict_classes(TestDataReshaped)
OutputCNN['PassengerId'] = td_merged.PassengerId.astype(int)
```


```python
OutputNN = testing_data.copy()
OutputNN["Survived"] = modelNN.predict_classes(testing_data.to_numpy())
OutputNN['PassengerId'] = td_merged.PassengerId.astype(int)
```


```python
OutputXGB = testing_data.copy()
OutputXGB["Survived"] = np.array(
    modelXGB.predict(
        xgb.DMatrix(testing_data.to_numpy(), feature_names=testing_data.columns.values)
    ).round()
    ,dtype=int)
OutputXGB['PassengerId'] = td_merged.PassengerId.astype(int)
```


```python
OutputKNN = testing_data.copy()
OutputKNN["Survived"] = modelKNN.predict(testing_data.to_numpy())
OutputKNN.Survived = OutputKNN.Survived.astype(int)
OutputKNN['PassengerId'] = td_merged.PassengerId.astype(int)
```


```python
OutputLGB = testing_data.copy()
OutputLGB["Survived"] = np.array(bst.predict(testing_data.to_numpy(), num_iteration=bst.best_iteration).round(),dtype=int)
OutputLGB['PassengerId'] = td_merged.PassengerId.astype(int)
```

### Save predictions to file


```python
OutputLGB[['PassengerId', 'Survived']].to_csv('output/LGB.csv', index=False)
OutputXGB[['PassengerId', 'Survived']].to_csv('output/XGB.csv', index=False)
OutputCNN[['PassengerId', 'Survived']].to_csv('output/CNN.csv', index=False)
OutputNN[['PassengerId', 'Survived']].to_csv('output/NN.csv', index=False)
OutputKNN[['PassengerId', 'Survived']].to_csv('output/KNN.csv', index=False)
```
