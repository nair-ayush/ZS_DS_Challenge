#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('./data.csv', index_col=0)


# In[3]:


df.shape


# In[87]:


testM = df.loc[(df.is_goal.isna()) & (~df.shot_id_number.isna()),:]
train = df.loc[~df.is_goal.isna(),:]


# In[ ]:


print("Train shape: ", train.shape)
print("Test shape:", test.shape)


# In[ ]:


train.head()


# ## Cleaning

# In[ ]:


train.columns


# In[5]:


train = train.drop(['match_event_id','game_season','shot_id_number','match_id','team_id','remaining_min.1','remaining_sec.1','knockout_match.1','power_of_shot.1','distance_of_shot.1','team_name'],axis=1)


# In[6]:


from sklearn.impute import SimpleImputer


# In[7]:


meanImputer = SimpleImputer(np.nan, strategy='mean')
medianImputer = SimpleImputer(np.nan, strategy='median')
frequentImputer = SimpleImputer(np.nan, strategy='most_frequent')


# In[9]:


train['is_shot'] = train.type_of_shot.apply(lambda x: 0 if x is np.nan else 1)
train['is_combined_shot'] = train.type_of_combined_shot.apply(lambda x: 0 if x is np.nan else 1)

train.type_of_combined_shot = train.type_of_combined_shot.fillna("000")
train.type_of_shot = train.type_of_shot.fillna("00")

train.type_of_combined_shot = train.type_of_combined_shot.apply(lambda x: int(str(x)[-1]))
train.type_of_shot = train.type_of_shot.apply(lambda x: int(str(x)[-1]))

train['shot_type'] = train.type_of_combined_shot + train.type_of_shot

train['lat'] = train['lat/lng'].apply(lambda x: float(str(x).split(", ")[0]))
train['long'] = train['lat/lng'].apply(lambda x: float(str(x).split(", ")[-1]))

train = train.drop(['type_of_shot','type_of_combined_shot','lat/lng'], axis=1)


# In[10]:


medianColumns = ['remaining_min','knockout_match','lat','long', 'power_of_shot']
meanColumns = ['location_x','location_y','remaining_sec','distance_of_shot']
frequentColumns = ['area_of_shot','shot_basics','range_of_shot','date_of_game','home/away']
print(len(medianColumns + meanColumns + frequentColumns))


# In[13]:


meanImputer.fit(train.loc[:,meanColumns])
medianImputer.fit(train.loc[:,medianColumns])
frequentImputer.fit(train.loc[:,frequentColumns])


# In[14]:


trainImputedMean = meanImputer.transform(train.loc[:,meanColumns])
trainImputedMedian = medianImputer.transform(train.loc[:,medianColumns])
trainImputedFrequent = frequentImputer.transform(train.loc[:, frequentColumns])


# In[24]:


yTrain = train['is_goal']


# In[17]:


prefinalTrain = pd.DataFrame(np.concatenate([trainImputedMean, trainImputedMedian, train[['is_shot','is_combined_shot','shot_type']].values], axis=1))


# In[18]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[22]:


for i in range(5):
    trainImputedFrequent[:,i] = le.fit_transform(trainImputedFrequent[:,i])

finalTrain = pd.concat([prefinalTrain, pd.DataFrame(trainImputedFrequent)], axis=1)


# ## Modelling

# In[23]:


from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, tree
import xgboost

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split (finalTrain, yTrain, test_size = 0.20, random_state=42)


# In[ ]:


pipelines = []
pipelines.append(('ScaledSVC', Pipeline([('Scaler', StandardScaler()),('SVC', svm.SVC())])))
pipelines.append(('ScaledDT', Pipeline([('Scaler', StandardScaler()),('DT', tree.DecisionTreeClassifier())])))
pipelines.append(('ScaledXGB', Pipeline([('Scaler', StandardScaler()),('XGB',xgboost.XGBClassifier())])))
pipelines.append(('ScaledRF', Pipeline([('Scaler', StandardScaler()),('RF', RandomForestClassifier())])))



results = []
names = []
for name, model in pipelines:
    kfold = KFold(n_splits=5, random_state=21)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='neg_mean_absolute_error')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# ## Hyperparameter Tuning

# In[ ]:


from sklearn.model_selection import GridSearchCV

scaler = StandardScaler().fit(X_train)
rescaledX = scaler.transform(X_train)
param_grid = dict(n_estimators=np.array([50,100,200,300,400]), max_depth=np.array([5,9,14]))
model = xgboost.XGBClassifier(random_state=21)
kfold = KFold(n_splits=5, random_state=21)
grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_mean_absolute_error', cv=kfold)
grid_result = grid.fit(rescaledX, Y_train)


# In[ ]:


means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# ## Making predictions

# In[88]:


yTest = testM['is_goal']
test = test.drop(['match_event_id','game_season','shot_id_number','match_id','team_id','remaining_min.1','remaining_sec.1','knockout_match.1','power_of_shot.1','distance_of_shot.1','team_name','is_goal'],axis=1)
test['is_shot'] = test.type_of_shot.apply(lambda x: 0 if x is np.nan else 1)
test['is_combined_shot'] = test.type_of_combined_shot.apply(lambda x: 0 if x is np.nan else 1)

test.type_of_combined_shot = test.type_of_combined_shot.fillna("000")
test.type_of_shot = test.type_of_shot.fillna("00")

test.type_of_combined_shot = test.type_of_combined_shot.apply(lambda x: int(str(x)[-1]))

test.type_of_shot = test.type_of_shot.apply(lambda x: int(str(x)[-1]))

test['shot_type'] = test.type_of_combined_shot + test.type_of_shot

test['lat'] = test['lat/lng'].apply(lambda x: float(str(x).split(", ")[0]))
test['long'] = test['lat/lng'].apply(lambda x: float(str(x).split(", ")[-1]))
test.drop(['lat/lng','type_of_shot','type_of_combined_shot'], axis=1, inplace=True)


# In[89]:


testImputedMean = meanImputer.transform(test.loc[:,meanColumns])
testImputedMedian = medianImputer.transform(test.loc[:,medianColumns])
testImputedFrequent = frequentImputer.transform(test.loc[:,frequentColumns])


# In[90]:


prefinalTest = pd.DataFrame(np.concatenate([testImputedMean, testImputedMedian, test[['is_shot','is_combined_shot','shot_type']].values], axis=1))


# In[91]:


for i in range(5):
    testImputedFrequent[:,i] = le.fit_transform(testImputedFrequent[:,i])

finalTest = pd.concat([prefinalTest, pd.DataFrame(testImputedFrequent)], axis=1)


# In[92]:


finalTrain.shape


# In[93]:


from sklearn.metrics import mean_absolute_error

scaler = StandardScaler().fit(finalTrain)
rescaled_final = scaler.transform(finalTrain)
model = xgboost.XGBClassifier(random_state=21, n_estimators=50, max_depth=5)
model.fit(rescaled_final, yTrain)

# transform the validation dataset
rescaled_X_test = scaler.transform(finalTest)
predictions = model.predict_proba(rescaled_X_test)[:,1]


# In[69]:


submission = pd.read_csv('sample_submission.csv')


# In[95]:


idx = testM.loc[:,'shot_id_number']


# In[98]:


predictions.shape


# In[103]:


pred = pd.Series(predictions)


# In[104]:


pred.index = idx.index


# In[96]:


idx.shape


# In[119]:


sub.to_csv('ayush_nair_082898_prediction_1.csv', index=False)


# In[118]:


sub = pd.concat([idx, pred], axis=1, ignore_index=True)
sub.columns = ['shot_id_number','is_goal']
sub.shot_id_number = sub.shot_id_number.astype('int64')

