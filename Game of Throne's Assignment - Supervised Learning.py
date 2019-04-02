# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 20:58:26 2019

@author: Destini B
"""

"""Machine Learning - Game of Thrones Surpervised Learning"""
##############################################################################
#Game of Thrones
##############################################################################
# Loading Libraries
import pandas as pd
import seaborn as sns
import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# Loading the Game of Thrones Dataset
file = 'GOT_character_predictions.xlsx'
game_of_thrones = pd.read_excel(file)

##############################################################################
#Exploring and modifying the dataset
##############################################################################

#Describing the dataset
game_of_thrones.shape #how many rows and columns
game_of_thrones.info() #how many are not null and there type
game_of_thrones.describe().round(2)  #summary of each variable

"""The dataset is composed of 1946 rows and 26 columns with float,int, and 
    object data types"""

#Checking for missing values
missing_got = game_of_thrones.isnull().sum()
total_got = game_of_thrones.count()
missing_ratio = missing_got / total_got
missing_ratio.round(2)

"""From our missing value ratio we can conclude 92% of mother, 74% of father,
   83% of heir, and the corresponding IS alive columns are missing. Therefore,
   it would be a wise decision to drop these variables as they would add no value
   at this point"""


#Flagging missing values
for col in game_of_thrones:
    if game_of_thrones[col].isnull().any():
        game_of_thrones['m_'+col] = game_of_thrones[col].isnull().astype(int)
        
"""The missing values have been flagged and there are now an additional 13 
columns indicating missing values with the notation m_ column name."""
              
#Changing Values in the male column from male to female
female_to_male = ['Roslin Frey','Lyanna Mormont', 'Becca', 'Kyra', 'Rhaella Targaryen']

game_of_thrones.loc[game_of_thrones.name.isin(female_to_male), "male"] =0

#Changing Values in the male column from female to male
male_to_female = ['Osmund Frey', 'Walder Frey', 'Humfrey Wagstaff', 'Symond Templeton',
                  'Timett (father)', 'Lem (Standfast)', 'Luthor Tyrell', 'Narbert',
                  'Raymund Terrell', ' Kyle (brotherhood)', 'Humfrey Swyft','Harmond Umber',
                  'Lew (guard)','Manfrey Martell']

game_of_thrones.loc[game_of_thrones.name.isin(male_to_female), "male"] =1
       
game_of_thrones['culture']=np.where(game_of_thrones['culture']=='Andal','Andals', game_of_thrones['culture'])
"V"

#Imputing the missing values for each column
df_game_of_thrones = pd.DataFrame.copy(game_of_thrones)

df_game_of_thrones['title'] = df_game_of_thrones['title'].fillna('Unknown')
title_dummies=pd.get_dummies((df_game_of_thrones['title']), drop_first = True)

df_game_of_thrones['culture'] = df_game_of_thrones['culture'].fillna('Unknown')
culture_dummies = pd.get_dummies((df_game_of_thrones['culture']), drop_first = True)

df_game_of_thrones['dateOfBirth'] = df_game_of_thrones['dateOfBirth'].fillna(pd.np.mean(df_game_of_thrones['dateOfBirth'])).astype(int)

df_game_of_thrones['house'] = df_game_of_thrones['house'].fillna('Unknown')
house_dummies=pd.get_dummies((df_game_of_thrones['house']), drop_first = True)

df_game_of_thrones['age'][df_game_of_thrones['age'] < 0] = 0
df_game_of_thrones['age'].sort_values()
df_game_of_thrones['age'] = df_game_of_thrones['age'].fillna(pd.np.mean(df_game_of_thrones['age']))

"""Created dummy variables for the culture and housing columns through the use 
   of one hot encoding to turn a categorical variable into a binary value to
   use for predicting."""

"""Dropping those variables that have too many missing values: mother, father, 
   heir, and their corresponding isAlive columns.
 """

df_dropped = df_game_of_thrones.loc[:, ['mother',
                                        'father',
                                        'heir',
                                        'spouse',
                                        'isAliveMother',
                                        'isAliveFather',
                                        'isAliveHeir',
                                        'isAliveSpouse']]

#Prepping the dataset to perform the train test split
game_of_thrones_data_1 = df_game_of_thrones.loc[:, ['male',
                                                    'book1_A_Game_Of_Thrones',
                                                    'book2_A_Clash_Of_Kings',
                                                    'book3_A_Storm_Of_Swords',
                                                    'book4_A_Feast_For_Crows',
                                                    'book5_A_Dance_with_Dragons',
                                                    'isMarried',
                                                    'isNoble',
                                                    'isAlive',
                                                    'age',
                                                    'dateOfBirth',
                                                    'popularity',
                                                    'm_title',
                                                    'm_culture',
                                                    'm_house',
                                                    ]]

#Concatenanting the dummy variables into the dataset 
game_of_thrones_data = pd.concat([game_of_thrones_data_1.iloc[:,:],
                                  house_dummies,
                                  culture_dummies],
                                  axis = 1)


game_of_thrones_target = df_game_of_thrones['isAlive']

##############################################################################
#Train Test Split
##############################################################################
"""Spliting the data set into a train and testing dataset to utilize for future 
testing of models""" #Need to re look at this
X_train, X_test, y_train, y_test = train_test_split(
            game_of_thrones_data,
            game_of_thrones_target,
            test_size = 0.1,
            random_state = 508)

#Training Set
print(X_train.shape)
print(y_train.shape)

""" The training set consists of 1751 rows and 425 columns (X_test) and (y_test)
    has 1459 rows and 1 column."""

# Testing set
print(X_test.shape)
print(y_test.shape)

""" The testing set consists of 195 rows and 425 columns (X_test) and (y_test)
    has 195 rows and 1 column."""

#############################################################################
#KNN Neighbors
#############################################################################
training_accuracy = []
test_accuracy = []

neighbors_range = range(1, 51)

for n_neighbors in neighbors_range:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_train, y_train.values.ravel())
    
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))   
    print(test_accuracy)
    
fig, ax = plt.subplots(figsize=(12,9))
plt.plot(neighbors_range, training_accuracy, label = "training accuracy")
plt.plot(neighbors_range, test_accuracy, label = "test accuracy")
plt.ylabel("Accuracy")
plt.xlabel("n_neighbors")
plt.legend()
plt.show()

#Finding the optimal K Score
knn_clf = KNeighborsClassifier(n_neighbors = 3)
# Fitting the model based on the training data
knn_clf_fit = knn_clf.fit(X_train, y_train)
# Scoring the model
y_score = knn_clf.score(X_test, y_test)
# The score is directly comparable to R-Square
print(y_score.round(3))
"""The optimal number of neighbors is 3 with a y score of .964."""


###############################################################################
# Cross Validation with K-folds
###############################################################################
cv_knn_3 = cross_val_score(knn_clf.fit,
                           game_of_thrones_data,
                           game_of_thrones_target,
                           cv = 3)


print(cv_knn_3)


print(pd.np.mean(cv_knn_3).round(3))

print('\nAverage: ',
      pd.np.mean(cv_knn_3).round(3),
      '\nMinimum: ',
      min(cv_knn_3).round(3),
      '\nMaximum: ',
      max(cv_knn_3).round(3))

"""The average cross validation score is .945, Minimum is .941, and Maximum is .951"""

############################################################################
#AUC Score
########################################################################## 
KNN_pred_probabilities = knn_clf_fit.predict_proba(X_test)

KNN_auc_score = roc_auc_score(y_test, KNN_pred_probabilities[:, 1]).round(3)
KNN_auc_score
"""The resulting AUC score is .933 for the KNN Neighbors"""
     
#############################################################################
#Logistic Regression - Hyperparameter 
#############################################################################
logreg_100 = LogisticRegression(C = 100,
                                solver = 'lbfgs')


logreg_100_fit = logreg_100.fit(X_train, y_train)


logreg_pred = logreg_100_fit.predict(X_test)


# Let's compare the testing score to the training score.
print('Training Score', logreg_100_fit.score(X_train, y_train).round(4))
print('Testing Score:', logreg_100_fit.score(X_test, y_test).round(4))

#############################################################################
#AUC Score  = 1
############################################################################

log_pred_probabilities = logreg_100_fit.predict_proba(X_test)

log_auc_score = roc_auc_score(y_test, log_pred_probabilities[:, 1]).round(3)
log_auc_score

"""The prediction score is 1, resulting in the auc score being 1."""
#############################################################################
#Random Forest
#############################################################################

# Full forest using gini
full_forest_gini = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'gini',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Full forest using entropy
full_forest_entropy = RandomForestClassifier(n_estimators = 500,
                                     criterion = 'entropy',
                                     max_depth = None,
                                     min_samples_leaf = 15,
                                     bootstrap = True,
                                     warm_start = False,
                                     random_state = 508)

# Fitting the models
full_gini_fit = full_forest_gini.fit(X_train, y_train)
full_entropy_fit = full_forest_entropy.fit(X_train, y_train)


pd.DataFrame(full_gini_fit.predict(X_test), full_entropy_fit.predict(X_test))
full_gini_fit.predict(X_test).sum() == full_entropy_fit.predict(X_test).sum()

# Scoring the gini model
print('Training Score', full_gini_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_gini_fit.score(X_test, y_test).round(4))

""" Training Score is .7601 and testing score =.7487"""

# Scoring the entropy model
print('Training Score', full_entropy_fit.score(X_train, y_train).round(4))
print('Testing Score:', full_entropy_fit.score(X_test, y_test).round(4))

"""Training score is .7613 and testing score is .7487"""

# Saving score objects
gini_full_train = full_gini_fit.score(X_train, y_train)
gini_full_test  = full_gini_fit.score(X_test, y_test)

entropy_full_train = full_entropy_fit.score(X_train, y_train)
entropy_full_test  = full_entropy_fit.score(X_test, y_test)

"""Due to their being too many variables, when utilizing the feature importance
plot the graphic isn't visually appealing and I chose not to use it.""" 

##############################################################################
#AUC Scores 
#############################################################################
gini_pred_probabilities = full_gini_fit.predict_proba(X_test)

gini_auc_score = roc_auc_score(y_test, gini_pred_probabilities[:, 1]).round(3)
gini_auc_score
"""The gini AUC score = 1"""

entropy_pred_probabilities = full_entropy_fit.predict_proba(X_test)

entropy_auc_score = roc_auc_score(y_test, entropy_pred_probabilities[:, 1]).round(3)
entropy_auc_score
"""The entropy_auc_score = 1"""

##############################################################################
#GBM
##############################################################################
# Creating a hyperparameter grid
learn_space = pd.np.arange(0.01, 2.01, 0.05)
estimator_space = pd.np.arange(50, 1000, 50)
depth_space = pd.np.arange(1, 10)
leaf_space = pd.np.arange(1, 150, 15)
criterion_space = ['friedman_mse', 'mse', 'mae']


param_grid = {'learning_rate' : learn_space,
              'n_estimators' : estimator_space,
              'max_depth' : depth_space,
              'min_samples_leaf' : leaf_space,
              'criterion' : criterion_space}

# Building the model object one more time
gbm_grid = GradientBoostingRegressor(random_state = 508)

# Creating a GridSearchCV object
gbm_grid_cv = RandomizedSearchCV(estimator = gbm_grid,
                                 param_distributions = param_grid,
                                 n_iter = 50,
                                 scoring = None,
                                 cv = 3,
                                 random_state = 508)
# Fit it to the training data
gbm_basic_fit = gbm_grid_cv.fit(X_train, y_train)

# Print the optimal parameters and best score
print("Tuned GBM Parameter:", gbm_grid_cv.best_params_)
print("Tuned GBM Accuracy:", gbm_grid_cv.best_score_.round(4))
#############################################################################
#AUC Score
############################################################################
GBM_pred_probabilities = gbm_basic_fit.predict_proba(X_test)

GBM_auc_score = roc_auc_score(y_test, GBM_pred_probabilities[:, 1]).round(3)
GBM_auc_score

#############################################################################


