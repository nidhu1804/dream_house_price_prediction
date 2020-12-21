#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import tree
import xgboost as xgb
from sklearn.ensemble import BaggingRegressor
import numpy as np 
import pandas as pd


# In[13]:


train = pd.read_csv('train_houseprice.csv')
test = pd.read_csv('test_houseprice.csv')


# In[14]:


train.head()


# In[15]:


sum(train.isna().sum())


# In[16]:


sum(test.isna().sum())


# In[17]:


for name in train.columns:
    x = train[name].isna().sum()
    if x > 0:
        val_list = np.random.choice(train.groupby(name).count().index, x, p=train.groupby(name).count()['Id'].values /sum(train.groupby(name).count()['Id'].values))
        train.loc[train[name].isna(), name] = val_list


# In[18]:


for name in test.columns:
    x = test[name].isna().sum()
    if x > 0:
        val_list = np.random.choice(test.groupby(name).count().index, x, p=test.groupby(name).count()['Id'].values /sum(test.groupby(name).count()['Id'].values))
        test.loc[test[name].isna(), name] = val_list


# In[19]:


sum(train.isna().sum())
sum(test.isna().sum())


# In[20]:


train_df = train.drop('SalePrice',axis = 1)
data = pd.concat([train_df,test])
le = preprocessing.LabelEncoder()
for name in data.columns:
    if data[name].dtypes == "O":
        print(name)
        data[name] = data[name].astype(str)
        train[name] = train[name].astype(str)
        test[name] = test[name].astype(str)
        le.fit(data[name])
        train[name] = le.transform(train[name])
        test[name] = le.transform(test[name])


# In[21]:


for name in test.columns:
    if test[name].dtypes == "O":
        test[name] = test[name].to_string()
        le.fit(test[name])
        test[name] = le.transform(test[name])


# In[22]:


X = train.drop('SalePrice',axis = 1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[23]:


regr = RandomForestRegressor(max_depth=2, random_state=0)
regr.fit(X_train, y_train)
predictions = regr.predict(X_test)


# In[24]:


mean_squared_error(predictions, y_test)


# In[25]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[26]:


from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents_train = pca.fit_transform(X)
principalComponents_test = pca.fit_transform(test)
sum(pca.explained_variance_ratio_)


# In[27]:


train['component_1'] = [i[0] for i in principalComponents_train]
train['component_2'] = [i[1] for i in principalComponents_train]
train['component_3'] = [i[2] for i in principalComponents_train]
test['component_1'] = [i[0] for i in principalComponents_test]
test['component_2'] = [i[1] for i in principalComponents_test]
test['component_3'] = [i[2] for i in principalComponents_test]


# In[28]:


X = train.drop('SalePrice',axis = 1)
y = train['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
regr = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)
regr.fit(X, y)
predictions = regr.predict(X)
mean_squared_error(predictions, y)


# In[29]:


model_1 = RandomForestRegressor(n_estimators = 400,min_samples_split = 2,min_samples_leaf = 1,max_features= 'sqrt',max_depth =None,bootstrap= False)
model_1.fit(X, y)
predict_1 = model_1.predict(X)
model_2= linear_model.Ridge()
model_2.fit(X,y)
predict_2 =model_2.predict(X)
model_3 =KNeighborsRegressor(10,weights='uniform')
model_3.fit(X,y)
predict_3 = model_3.predict(X)
model_4 = linear_model.BayesianRidge()
model_4.fit(X,y)
predict_4 =model_4.predict(X)
model_5 = tree.DecisionTreeRegressor(max_depth=1)
model_5.fit(X,y)
predict_5 =model_5.predict(X)
model_6= svm.SVR(C=1.0, epsilon=0.2)
model_6.fit(X,y)
predict_6 = model_6.predict(X)
model_7 = xgb.XGBRegressor()
model_7.fit(X,y)
predict_7 = model_7.predict(X)


# In[30]:


final_df = pd.DataFrame()
final_df['SalePrice'] = y
final_df['RandomForest'] = predict_1
final_df['Ridge'] = predict_2
final_df['Kneighboors'] = predict_3
final_df['BayesianRidge'] = predict_4
final_df['DecisionTreeRegressor'] = predict_5
final_df['Svm'] = predict_6
final_df['XGBoost'] = predict_7


# In[31]:


print(mean_squared_error(final_df['SalePrice'], predict_1))
print(mean_squared_error(final_df['SalePrice'], predict_2))
print(mean_squared_error(final_df['SalePrice'], predict_3))
print(mean_squared_error(final_df['SalePrice'], predict_4))
print(mean_squared_error(final_df['SalePrice'], predict_5))
print(mean_squared_error(final_df['SalePrice'], predict_6))
print(mean_squared_error(final_df['SalePrice'], predict_7))


# In[32]:


X_final = final_df.drop('SalePrice',axis = 1)
y_final = final_df['SalePrice']
model_last = RandomForestRegressor()
model_last.fit(X_final, y_final)
predict_final = model_last.predict(X_final)
final_dt = RandomForestRegressor()                   
model_last = BaggingRegressor(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)
model_last.fit(X_final, y_final)
predict_final = model_last.predict(X_final)
acc_oob = model_last.oob_score_
print(acc_oob)


# In[33]:


mean_squared_error(predict_final, y_final)


# In[34]:


test_predictions_1 = model_1.predict(test)
test_predictions_2 = model_2.predict(test)
test_predictions_3 = model_3.predict(test)
test_predictions_4 = model_4.predict(test)
test_predictions_5 = model_5.predict(test)
test_predictions_6 = model_6.predict(test)
test_predictions_7 = model_7.predict(test)


# In[35]:


test_final_df = pd.DataFrame()
test_final_df['RandomForest'] = test_predictions_1
test_final_df['Ridge'] = test_predictions_2
test_final_df['Kneighboors'] = test_predictions_3
test_final_df['BayesianRidge'] = test_predictions_4
test_final_df['DecisionTreeRegressor'] = test_predictions_5
test_final_df['Svm'] = test_predictions_6
test_final_df['XGBoost'] = test_predictions_7


# In[36]:


last_predictions = model_last.predict(test_final_df)


# In[46]:


submission = pd.read_csv('sample_submission.csv')


# In[47]:


submission['SalePrice'] = last_predictions


# In[48]:


submission.to_csv('D:/ML/HOUSE PRICE PREDICTION/pt2/sample_submission.csv',index = False)


# In[ ]:




