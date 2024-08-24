#!/usr/bin/env python
# coding: utf-8

# # Multiple_Linear_Regression_Bike_Sharing_Assignment

#     Submitted by Anjali Jain

# #### Problem Statement:
# 
# A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario.In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.
# 
# ***The company wants to know:***
# 
# - Which variables are significant in predicting the demand for shared bikes.
# - How well those variables describe the bike demands
# 
# ***Goal:***
# - Develop a model to find the variables which are significant the demand for shared bikes with the available independent variables.
# - It will be used by the management to understand and manipulate the business strategy to meet the demand levels and meet the customer's expectations.

# ## Step1: Importing Libraries

# In[54]:


#imporitng the required packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')


# # Step2: Reading Dataset and Understanding Data

# In[55]:


#Read the csv file using pandas
bikeSharing_df = pd.read_csv('day.csv')


# In[56]:


#Inspect first few rows
bikeSharing_df.head()


# In[57]:


#check the shape
bikeSharing_df.shape


# In[58]:


#check dataframe for null and datatype 
bikeSharing_df.info()


# In[59]:


#check the details of numeriacl data
bikeSharing_df.describe()


# In[60]:


#check the columns of data
bikeSharing_df.columns


# In[61]:


#check the size of data
bikeSharing_df.size


# In[62]:


#check the datatypes of data
bikeSharing_df.dtypes


# In[63]:


#check the axes of data
bikeSharing_df.axes


# In[64]:


#check the dimensions of data
bikeSharing_df.ndim


# In[65]:


#check the values of data
bikeSharing_df.values


# #### There are 730 rows and 16 columns in the data set. There are no null values in any of the columns.

# # Step3: Cleaning Data

# ### 3.1 Drop columns that are not useful for analysis

# - `instant` is just a row instance identifier.
# - `dteday` is removed as we have some of date features like `mnth` and `year` and `weekday` already in other columns and also for this analysis we will not consider day to day trend in demand for bikes.
# - `casual` and `registered` variables are not available at the time of prediction and also these describe the target variable `cnt` in a very trivial way `target = casual + registered`, which leads to data leakage.

# In[66]:


#Before dropping date, let us introduce a days_old variable which indicates how old is the business
bikeSharing_df['days_old'] = (pd.to_datetime(bikeSharing_df['dteday'],format= '%d-%m-%Y') - pd.to_datetime('01-01-2018',format= '%d-%m-%Y')).dt.days


# In[67]:


#Inspect once
bikeSharing_df.head()


# In[68]:


# Droping instant column as it is index column which has nothing to do with target
bikeSharing_df.drop(['instant'], axis = 1, inplace = True)

# Dropping dteday as we have already have month and weekday columns to work with
bikeSharing_df.drop(['dteday'], axis = 1, inplace = True)

# Dropping casual and registered columnsa as as we have cnt column which is sum of the both that is the target column

bikeSharing_df.drop(['casual'], axis = 1, inplace = True)
bikeSharing_df.drop(['registered'], axis = 1, inplace = True)


# In[69]:


#Inspect data frame after dropping
bikeSharing_df.head()


# In[70]:


bikeSharing_df.info()


# In[71]:


bikeSharing_df.season.value_counts()


# In[72]:


bikeSharing_df.weathersit.value_counts()


# In[73]:


bikeSharing_df.corr()


# - we can see that features like `season, mnth, weekday and  weathersit` are integers although they should be non-numerical categories.

# ### 3.2 Handle Missing values

# #### As we have already seen there are no missing values. However, let us verify it again

# In[74]:


#Print null counts by column
bikeSharing_df.isnull().sum()


# #### _`Inference`_: There are no null values.

# ### 3.3 Handle Outliers

# In[75]:


### Handle Outliers
bikeSharing_df.columns


# In[76]:


#Print number of unique values in all columns
bikeSharing_df.nunique()


# In[77]:


# Draw box plots for indepent variables with continuous values
cols = ['temp', 'atemp', 'hum', 'windspeed']
plt.figure(figsize=(18,4))

i = 1
for col in cols:
    plt.subplot(1,4,i)
    sns.boxplot(y=col, data=bikeSharing_df)
    i+=1


# #### From these plots, we can see there are no outliers to be handled. We are good with not having any outliers in the data set

# # 4. EDA

# #### 4.1 Convert season and  weathersit to categorical types

# In[78]:


bikeSharing_df.season.replace({1:"spring", 2:"summer", 3:"fall", 4:"winter"},inplace = True)

bikeSharing_df.weathersit.replace({1:'good',2:'moderate',3:'bad',4:'severe'},inplace = True)

bikeSharing_df.mnth = bikeSharing_df.mnth.replace({1: 'jan',2: 'feb',3: 'mar',4: 'apr',5: 'may',6: 'jun',
                  7: 'jul',8: 'aug',9: 'sept',10: 'oct',11: 'nov',12: 'dec'})

bikeSharing_df.weekday = bikeSharing_df.weekday.replace({0: 'sun',1: 'mon',2: 'tue',3: 'wed',4: 'thu',5: 'fri',6: 'sat'})
bikeSharing_df.head()


# #### 4.2 Draw pair Plots to check the linear relationship

# In[79]:


#Draw pairplots for continuous numeric variables using seaborn
plt.figure(figsize = (15,30))
sns.pairplot(data=bikeSharing_df,vars=['cnt', 'temp', 'atemp', 'hum','windspeed'])
plt.show()


# #### _`Inference`_: 
# - Looks like the temp and atemp has the highest corelation with the target variable cnt
# - temp and atemp are highly co-related with each other
# #### As seen from the correlation map, output variable has a linear relationship with variables like temp, atemp. 

# #### 4.3 Visualising the Data to Find the Correlation between the Numerical Variable

# In[80]:


plt.figure(figsize=(20,15))
sns.pairplot(bikeSharing_df)
plt.show()


# In[81]:


# Checking continuous variables relationship with each other
sns.heatmap(bikeSharing_df[['temp','atemp','hum','windspeed','cnt']].corr(), cmap='BuGn', annot = True)
plt.show()


# #### Here we see that temp and atemp has correlation more than .99 means almost 1 (highly correlated) and atemp seems to be derived from temp so atemp field can be dropped here only

# In[82]:


#Correlations for numeric variables
cor=bikeSharing_df.corr()
sns.heatmap(cor, cmap="YlGnBu", annot = True)
plt.show()


# #### 4.4 Draw Heatmap of correlation between variables

# In[83]:


#Calculate Correlation
corr = bikeSharing_df.corr()
plt.figure(figsize=(25,10))

#Draw Heatmap of correlation
sns.heatmap(corr,annot=True, cmap='YlGnBu' )
plt.show()


# #### From the correlation map, temp, atemp and days_old seems to be highly correlated and only should variable can be considered for the model. However let us elminate it based on the Variance Inflation Factor later during the model building.
# #### We also see Target variable has a linear relationship with some of the  indeptendent variables. Good sign for building a linear regression Model.

# #### 4.5 Analysing Categorical Variabels with target variables 

# In[84]:


# Boxplot for categorical variables to see demands
vars_cat = ['season','yr','mnth','holiday','weekday','workingday','weathersit']
plt.figure(figsize=(15, 15))
for i in enumerate(vars_cat):
    plt.subplot(3,3,i[0]+1)
    sns.boxplot(data=bikeSharing_df, x=i[1], y='cnt')
plt.show()


# #### _`Inference`_:
#     Here many insights can be drawn from the plots
# 
#     1. Season: 3:fall has highest demand for rental bikes
#     2. I see that demand for next year has grown
#     3. Demand is continuously growing each month till June. September month has highest demand. After September, demand is        decreasing
#     4. When there is a holiday, demand has decreased.
#     5. Weekday is not giving clear picture abount demand.
#     6. The clear weathershit has highest demand
#     7. During September, bike sharing is more. During the year end and beginning, it is less, could be due to extereme            weather conditions.

# In[85]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Temp",fontsize=16)
sns.regplot(data=bikeSharing_df,y="cnt",x="temp")
plt.xlabel("Temperature")
plt.show()


# #### _`Inference`_:
# - Demand for bikes is positively correlated to temp.
# - We can see that cnt is linearly increasing with temp indicating linear relation.

# In[86]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Hum",fontsize=16)
sns.regplot(data=bikeSharing_df,y="cnt",x="hum")
plt.xlabel("Humidity")
plt.show()


# #### _`Inference`_:
# - Hum is values are more scattered around.
# - Although we can see cnt decreasing with increase in humidity.

# In[87]:


plt.figure(figsize=(6,5),dpi=110)
plt.title("Cnt vs Windspeed",fontsize=16)
sns.regplot(data=bikeSharing_df,y="cnt",x="windspeed")
plt.show()


# #### _`Inference`_:
# - Windspeed is values are more scattered around.
# - Although we can see cnt decreasing with increase in windspeed.

# In[88]:


num_features = ["temp","atemp","hum","windspeed","cnt"]
plt.figure(figsize=(15,8),dpi=130)
plt.title("Correlation of numeric features",fontsize=16)
sns.heatmap(bikeSharing_df[num_features].corr(),annot= True,cmap="mako")
plt.show()


# #### `_Inference`_:
# - Temp and Atemp are highly correlated, we can take an action to remove one of them, but lets keep them for further analysis.
# - Temp and Atemp also have high correlation with cnt variable.

# In[89]:


bikeSharing_df.describe()


# # 5. Data Preparation for Linear Regression 

# #### 5.1 Create dummy variables for all categorical variables

# In[90]:


bikeSharing_df = pd.get_dummies(data=bikeSharing_df,columns=["season","mnth","weekday"],drop_first=True)
bikeSharing_df = pd.get_dummies(data=bikeSharing_df,columns=["weathersit"])


# - Dropping the first columns as (p-1) dummies can explain p categories.
# - In weathersit first column was not dropped so as to not lose the info about severe weather situation.

# In[91]:


#Print columns after creating dummies
bikeSharing_df.columns


# In[92]:


#Print few rows to inspect
bikeSharing_df.head()


# # 6. Model Building

# ## 5.1 Split Data into training and test

# In[93]:


# Checking shape before splitting
bikeSharing_df.shape


# In[94]:


#y to contain only target variable
y=bikeSharing_df.pop('cnt')

#X is all remainign variable also our independent variables
X=bikeSharing_df

#Train Test split with 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[95]:


#Inspect independent variables
X.head()


# In[96]:


# Checking shape and size for train and test
print(X_train.shape)
print(X_test.shape)


# ### 5.2 Feature Scaling continuous variables

# To make all features in same scale to interpret easily
# 
# Following columns are continous to be scaled
# temp,hum,windspeed

# In[97]:


# Importing required library
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler


# In[98]:


# Let us scale continuous variables
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Use Normalized scaler to scale
scaler = MinMaxScaler()

#Fit and transform training set only
X_train[num_vars] = scaler.fit_transform(X_train[num_vars])


# In[99]:


#Inspect stats fro Training set after scaling
X_train.describe()


# In[100]:


X_train.head()


# ## 5.3 Build a Model using RFE and Automated approach
# 
# #### Use RFE to eliminate some columns

# In[111]:


get_ipython().system('pip install RFE')


# In[112]:


# Build a Lienar Regression model using SKLearn for RFE
lr = LinearRegression()
lr.fit(X_train,y_train)


# In[114]:


#Cut down number of features to 15 using automated approach
rfe = RFE(estimator=lr, n_features_to_select=15)
rfe.fit(X_train,y_train)


# In[116]:


#Columns selected by RFE and their weights
list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# ## 5.4 Manual elimination

# ##### Function to build a model using statsmodel api

# In[117]:


#Function to build a model using statsmodel api - Takes the columns to be selected for model as a parameter
def build_model(cols):
    X_train_sm = sm.add_constant(X_train[cols])
    lm = sm.OLS(y_train, X_train_sm).fit()
    print(lm.summary())
    return lm


# ##### Function to calculate VIFs and print them

# In[118]:


#Function to calculate VIFs and print them -Takes the columns for which VIF to be calcualted as a parameter
def get_vif(cols):
    df1 = X_train[cols]
    vif = pd.DataFrame()
    vif['Features'] = df1.columns
    vif['VIF'] = [variance_inflation_factor(df1.values, i) for i in range(df1.shape[1])]
    vif['VIF'] = round(vif['VIF'],2)
    print(vif.sort_values(by='VIF',ascending=False))


# In[119]:


#Print Columns selected by RFE. We will start with these columns for manual elimination
X_train.columns[rfe.support_]


# In[120]:


# Features not selected by RFE
X_train.columns[~rfe.support_]


# In[121]:


# Taking 15 columns supported by RFE for regression
X_train_rfe = X_train[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']]


# In[122]:


X_train_rfe.shape


# # 6. Build Model 

# ### 6.1 Model 1 - Start with all variables selected by RFE

# In[57]:


#Selected columns for Model 1 - all columns selected by RFE
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']

build_model(cols)
get_vif(cols)


# In[58]:


# Checking correlation of features selected by RFE with target column. 
# Also to check impact of different features on target.
plt.figure(figsize = (15,10))
sns.heatmap(bikeSharing_df[['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jan', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']].corr(), cmap='GnBu', annot=True)
plt.show()


# ## Model 2

# In[59]:


# Dropping the variable mnth_jan as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 3

# #### All the columns have p-value > .05 so checking VIFs

# In[60]:


# Dropping the variable hum as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'holiday', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Module 4

# In[61]:


# Dropping the variable holiday as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_jul', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 5

# In[62]:


# Dropping the variable mnth_jul as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'temp', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 6

# In[63]:


# Dropping the variable temp as it has negative coefficient and is insignificant as it has high p-value
cols = ['yr', 'workingday', 'windspeed', 'season_spring',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 7

# In[64]:


## Trying to replace July with spring as both were highly correlated

cols = ['yr', 'workingday', 'windspeed', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 8

# In[65]:


## Trying to replace July with spring as both were highly correlated

cols = ['yr', 'workingday', 'windspeed', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 9

# In[66]:


# Removing windspeed with spring as windspeed was highly correlated with temp
cols = ['yr', 'workingday', 'season_spring', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sat',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 10

# In[67]:


# using the weekend "Sunday" which was dropped during RFE instead of Saturday.

cols = ['yr', 'workingday', 'season_spring', 'mnth_jul',
       'season_summer', 'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate']
build_model(cols)
get_vif(cols)


# ## Model 11

# In[68]:


# adding temp and removed 'season_summer' and 'workingday'
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#['yr', 'holiday','temp', 'spring','winter', 'July','September','Sunday','Light_Snow_Rain','Mist_Clody']
build_model(cols)
get_vif(cols)


# #### `_Inference`_
# Here VIF seems to be almost accepted. p-value for all the features is  almost 0.0 and R2 is 0.821 
# Let us select Model 11 as our final as it has all important statistics high (R-square, Adjusted R-squared and F-statistic), along with no insignificant variables and no multi coliinear (high VIF) variables. 
# Difference between R-squared and Adjusted R-squared values for this model is veryless, which also means that there are no additional parameters that can be removed from this model.

# In[69]:


#Build a model with all columns to select features automatically
def build_model_sk(X,y):
    lr1 = LinearRegression()
    lr1.fit(X,y)
    return lr1


# In[70]:


#Let us build the finalmodel using sklearn
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Build a model with above columns
lr = build_model_sk(X_train[cols],y_train)
print(lr.intercept_,lr.coef_)


# ## Step 7. Model Evaluation 
# ### 7.1 Residucal Analysis

# In[71]:


y_train_pred = lr.predict(X_train[cols])


# In[72]:


#Plot a histogram of the error terms
def plot_res_dist(act, pred):
    sns.distplot(act-pred)
    plt.title('Error Terms')
    plt.xlabel('Errors')


# In[73]:


plot_res_dist(y_train, y_train_pred)


# #### Errors are normally distribured here with mean 0. So everything seems to be fine

# In[74]:


# Actual vs Predicted
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train, color="blue")
plt.plot(c,y_train_pred, color="red")
plt.suptitle('Actual vs Predicted', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('Demands')
plt.show()


# #### Actual and Predicted result following almost the same pattern so this model seems ok

# In[75]:


# Error Terms
c = [i for i in range(0,len(X_train),1)]
plt.plot(c,y_train-y_train_pred)
plt.suptitle('Error Terms', fontsize = 15)
plt.xlabel('Index')
plt.ylabel('y_train-y_train_pred')
plt.show()


# #### Here,If we see the error terms are independent of each other.

# In[76]:


#Print R-squared Value
r2_score(y_train,y_train_pred)


# ### _`Inference`_
# R2 Same as we obtained for our final model

# ### 7.2 Linearity Check

# In[77]:


# scatter plot for the check
residual = (y_train - y_train_pred)
plt.scatter(y_train,residual)
plt.ylabel("y_train")
plt.xlabel("Residual")
plt.show()


# ### 7.3 Predict values for test data set

# In[78]:


#Scale variables in X_test
num_vars = ['temp','atemp','hum','windspeed','days_old']

#Test data to be transformed only, no fitting
X_test[num_vars] = scaler.transform(X_test[num_vars])


# In[79]:


#Columns from our final model
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

#Predict the values for test data
y_test_pred = lr.predict(X_test[cols])


# ### 7.4 R-Squared value for test predictions

# In[80]:


# Find out the R squared value between test and predicted test data sets.  
r2_score(y_test,y_test_pred)


# ### 7.5 Homoscedacity

# ##### _`Inference`_ 
# R2 value for predictions on test data (0.815) is almost same as R2 value of train data(0.818). This is a good R-squared value, hence we can see our model is performing good even on unseen data (test data)

# In[81]:


# Plotting y_test and y_test_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_test_pred)
fig.suptitle('y_test vs y_test_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_test_pred', fontsize = 16)


# ####  _`Inference`_
# We can observe that variance of the residuals (error terms) is constant across predictions.  i.e error term does not vary much as the value of the predictor variable changes.

# ### 7.8 Plot Test vs Predicted test values

# In[82]:


#Function to plot Actual vs Predicted
#Takes Actual and PRedicted values as input along with the scale and Title to indicate which data
def plot_act_pred(act,pred,scale,dataname):
    c = [i for i in range(1,scale,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act, color="blue", linewidth=2.5, linestyle="-")
    plt.plot(c,pred, color="red",  linewidth=2.5, linestyle="-")
    fig.suptitle('Actual and Predicted - '+dataname, fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                               # X-label
    plt.ylabel('Counts', fontsize=16)                               # Y-label


# In[83]:


#Plot Actual vs Predicted for Test Data
plot_act_pred(y_test,y_test_pred,len(y_test)+1,'Test Data')


# #### _`Inference`_
# As we can see predictions for test data is very close to actuals

# ### 7.9 Plot Error Terms for test data

# In[84]:


# Error terms
def plot_err_terms(act,pred):
    c = [i for i in range(1,220,1)]
    fig = plt.figure(figsize=(14,5))
    plt.plot(c,act-pred, color="blue", marker='o', linewidth=2.5, linestyle="")
    fig.suptitle('Error Terms', fontsize=20)              # Plot heading 
    plt.xlabel('Index', fontsize=18)                      # X-label
    plt.ylabel('Counts - Predicted Counts', fontsize=16)                # Y-label


# In[85]:


#Plot error terms for test data
plot_err_terms(y_test,y_test_pred)


# #### _`Inference`_
# As we can see the error terms are randomly distributed and there is no pattern which means the output is explained well by the model and there are no other parameters that can explain the model better.

# ### 8. Making Predictions

# In[86]:


# Checking data before scaling
bikeSharing_df.head()


# ### 8.1 Intrepretting the Model

# #### Let us go with interpretting the RFE with Manual model results as we give more importance to imputation

# In[87]:


#Let us rebuild the final model of manual + rfe approach using statsmodel to interpret it
cols = ['yr', 'season_spring', 'mnth_jul',
        'season_winter', 'mnth_sept', 'weekday_sun',
       'weathersit_bad', 'weathersit_moderate', 'temp']

lm = build_model(cols)


# ### Interepretation of results

# ### Analysing the above model, the comapany should focus on the following features:
# - Company should focus on expanding business during Spring.
# - Company should focus on expanding business during September.
# - Based on previous data it is expected to have a boom in number of users once situation comes back to normal, compared to 2019.
# - There would be less bookings during Light Snow or Rain, they could probably use this time to serive the bikes without having business impact.
# 
# #### Hence when the situation comes back to normal, the company should come up with new offers during spring when the weather is pleasant and also advertise a little for September as this is when business would be at its best.

# ### _`Conclusion`_
# Significant variables to predict the demand for shared bikes
# - holiday
# - temp
# - hum
# - windspeed
# - Season
# - months(January, July, September, November, December)
# - Year (2019)
# - Sunday
# - weathersit( Light Snow, Mist + Cloudy)
