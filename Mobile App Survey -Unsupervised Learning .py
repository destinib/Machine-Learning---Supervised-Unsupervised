# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 13:23:41 2019

@author: Destini B
"""
"""Machine Learning- Mobile App Survey Unsurpervised Learning Analysis"""

############################################################################
#Importing Libriaries
############################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans

# Setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

# Importing dataset
mobile_app_df = pd.read_excel('Mobile_App_Survey_Data.xlsx')

"""This dataset contains 1522 rows and 88 columns of survey data about mobile apps
    research from 1552 unique respondents"""

###########################################################################
#Exploring and Manipulating the Dataset
###########################################################################

#Renaming the columns with question names to identify better in analysis
mobile_app_df.rename(columns = {'q1'    : 'Age',
                                'q2r1'  : 'iPhone',
                                'q2r2'  : 'iPod_touch',
                                'q2r3'  : 'Android',
                                'q2r4'  : 'Blackberry',
                                'q2r5'  : 'Nokia',
                                'q2r6'  : 'Windows_Phone_Mobile',
                                'q2r7'  : 'HP_Palm_Web_OS',
                                'q2r8'  : 'Tablet',
                                'q2r9'  : 'Other_Smartphone',
                                'q2r10' : 'None_Q2',
                                'q4r1'  : 'Music_Sound_Apps',
                                'q4r2'  : 'TV_Checkin_Apps',
                                'q4r3'  : 'Entertainment_Apps',
                                'q4r4'  : 'TV_Show_Apps',
                                'q4r5'  : 'Gaming_Apps',
                                'q4r6'  : 'Social_Networking_Apps',
                                'q4r7'  : 'General_News_Apps',
                                'q4r8'  : 'Shopping_Apps',
                                'q4r9'  : 'Publication_News_Apps',
                                'q4r10' : 'Other_Apps',
                                'q4r11' : 'None_Q4',
                                'q11'   : 'Number_of_Smartphone_Apps',
                                'q12'   : '%_of_free_download',
                                'q13r1' : 'Facebook',
                                'q13r2' : 'Twitter',
                                'q13r3' : 'Myspace',
                                'q13r4' : 'Pandora_Radio',
                                'q13r5' : 'Vevo',
                                'q13r6' : 'YouTube',
                                'q13r7' : 'AOL_Radio',
                                'q13r8' : 'Last.fm',
                                'q13r9' : 'Yahoo_Entertainment_Music',
                                'q13r10': 'IMDB',
                                'q13r11': 'Linkedin',
                                'q13r12': 'Netflix',
                                'q24r1' : 'Uptodate_W/_Tech_Development',
                                'q24r2' : 'My_Advice_Buy_Tech_Elec_Products',
                                'q24r3' : 'Enjoy_Purch_New_Gadgets',
                                'q24r4' : 'Too_Much_Tech_Everyday',
                                'q24r5' : 'Enjoy_Using_Tech',
                                'q24r6' : 'Lookup_Webtools_Apps',
                                'q24r7' : 'Music_Important_MyLife',
                                'q24r8' : 'Learning_Fav_TV_Show',
                                'q24r9' : 'Too_Much_Info_Internet_Facebook',
                                'q24r10': 'Checkon_FriendsFam_SocialNetworkingSites',
                                'q24r11': 'Internet_Easier_Contact_FamFriends',
                                'q24r12': 'Internet_Easy_Avoid_FamFriends',
                                'q25r1' : 'Myself_Opinion_Leader',
                                'q25r2' : 'I_Stand_Out',
                                'q25r3' : 'Offer_Advice',
                                'q25r4' : 'Take_Lead_Decision_Making',
                                'q25r5' : 'First_Try_New_Things',
                                'q25r6' : 'Told_What_To_Do',
                                'q25r7' : 'Being_In_Control',
                                'q25r8' : 'Risk_Taker',
                                'q25r9' : 'Myself_Creative',
                                'q25r10': 'Optimistic_Person',
                                'q25r11': 'Very_Active_OntheGo',
                                'q25r12': 'Feel_Stretched_For_Time',
                                'q26r3' : 'Lookout_Bargin_Discounts_Deals',
                                'q26r4' : 'Enjoyment_Any_Kind_Shopping',
                                'q26r5' : 'Like_Package_Deals',
                                'q26r6' : 'Always_Shopping_Online',
                                'q26r7' : 'Prefer_Buy_Designer_Brands',
                                'q26r8' : 'Cant_Get_Enough_Apps',
                                'q26r9' : 'Coolness_Apps_Matter_Morethan_Number',
                                'q26r10': 'Showing_New_Apps',
                                'q26r11': 'Children_Impact_Apps_iDownload',
                                'q26r12': 'Worth_Spending_Extra_Extra_AppFeatures',
                                'q26r13': 'No_Point_Earning_Money_Not_Going_Spendit',
                                'q26r14': 'Influenced_WhatsHot_WhatsNot',
                                'q26r15': 'Buy_Brands_Reflect_My_Style',
                                'q26r16': 'Make_Impulse_Decisions',
                                'q26r17': 'Mobile_Phone_Sourceof_Entertainment',
                                'q26r18': 'Attracted_Luxury_Brands',
                                'q48'   : 'Education',
                                'q49'   : 'Marital_Status',
                                'q50r1' : 'No_Children',
                                'q50r2' : 'Children_Under_6',
                                'q50r3' : 'Children_6_to_12',
                                'q50r4' : 'Children_13_to_17',
                                'q50r5' : 'Children_18_older',
                                'q54'   : 'Race',
                                'q55'   :'Hispanic_Latino',
                                'q56'   : 'Household_Income',
                                'q57'   : 'Gender'},
                                inplace = True)

#Dropping the Demographic Information
survey_features_reduced = mobile_app_df.loc[ : ,['iPhone',
                                'iPod_touch',
                                'Android',
                                'Blackberry',
                                'Nokia',
                                'Windows_Phone_Mobile',
                                'HP_Palm_Web_OS',
                                'Tablet',
                                'Other_Smartphone',
                                'None_Q2',
                                'Music_Sound_Apps',
                                'TV_Checkin_Apps',
                                'Entertainment_Apps',
                                'TV_Show_Apps',
                                'Gaming_Apps',
                                'Social_Networking_Apps',
                                'General_News_Apps',
                                'Shopping_Apps',
                                'Publication_News_Apps',
                                'Other_Apps',
                                'None_Q4',
                                'Number_of_Smartphone_Apps',
                                '%_of_free_download',
                                'Facebook',
                                'Twitter',
                                'Myspace',
                                'Pandora_Radio',
                                'Vevo',
                                'YouTube',
                                'AOL_Radio',
                                'Last.fm',
                                'Yahoo_Entertainment_Music',
                                'IMDB',
                                'Linkedin',
                                'Netflix',
                                'Uptodate_W/_Tech_Development',
                                'My_Advice_Buy_Tech_Elec_Products',
                                'Enjoy_Purch_New_Gadgets',
                                'Too_Much_Tech_Everyday',
                                'Enjoy_Using_Tech',
                                'Lookup_Webtools_Apps',
                                'Music_Important_MyLife',
                                'Learning_Fav_TV_Show',
                                'Too_Much_Info_Internet_Facebook',
                                'Checkon_FriendsFam_SocialNetworkingSites',
                                'Internet_Easier_Contact_FamFriends',
                                'Internet_Easy_Avoid_FamFriends',
                                'Myself_Opinion_Leader',
                                'I_Stand_Out',
                                'Offer_Advice',
                                'Take_Lead_Decision_Making',
                                'First_Try_New_Things',
                                'Told_What_To_Do',
                                'Being_In_Control',
                                'Risk_Taker',
                                'Myself_Creative',
                                'Optimistic_Person',
                                'Very_Active_OntheGo',
                                'Feel_Stretched_For_Time',
                                'Lookout_Bargin_Discounts_Deals',
                                'Enjoyment_Any_Kind_Shopping',
                                'Like_Package_Deals',
                                'Always_Shopping_Online',
                                'Prefer_Buy_Designer_Brands',
                                'Cant_Get_Enough_Apps',
                                'Coolness_Apps_Matter_Morethan_Number',
                                'Showing_New_Apps',
                                'Children_Impact_Apps_iDownload',
                                'Worth_Spending_Extra_Extra_AppFeatures',
                                'No_Point_Earning_Money_Not_Going_Spendit',
                                'Influenced_WhatsHot_WhatsNot',
                                'Buy_Brands_Reflect_My_Style',
                                'Make_Impulse_Decisions',
                                'Mobile_Phone_Sourceof_Entertainment',
                                'Attracted_Luxury_Brands']]

"""The new dataframe consists of 1552 rows and 75 columns. The demographic 
questions were removed. This includes the following questions q1, q48, q49, q50,
q54, q55,q56, and q57."""

#############################################################################
#Prepping the dataset to perform PCA and KMeans Concepts
#############################################################################
# Scaling the new dataset to get equal variance
scaler = StandardScaler()

scaler.fit(survey_features_reduced)

X_scaled_reduced = scaler.transform(survey_features_reduced)

"""Because the features of this dataset are different, I utilized the Standard
Scaler function to improve the clustering by equating the variance"""

##############################################################################
#Principal Component Analysis - (PCA)
##############################################################################

#PCA without limiting the number of component
survey_pca_reduced = PCA(n_components = None,
                           random_state = 508)

survey_pca_reduced.fit(X_scaled_reduced)

X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)

"""Performed Principal Component Analysis on the reduced dataset"""


# Creating and Analyzing the scree plot 
fig, ax = plt.subplots(figsize=(12, 4))

features = range(survey_pca_reduced.n_components_)

plt.plot(features,
         survey_pca_reduced.explained_variance_ratio_,
         linewidth = 2,
         marker = 'o',
         markersize = 10,
         markeredgecolor = 'black',
         markerfacecolor = 'grey')

plt.title('Reduced Mobile App Survey Scree Plot')
plt.xlabel('PCA feature')
plt.ylabel('Explained Variance')
plt.xticks(features, size = 8, rotation ='90')
plt.show()
 
"""As seen in the scree plot, the inertia starts to consistently decrease around 
the 4th component. Therefore, I will choose the number of desired components as
4 and run PCA again."""


#Run PCA again with n_components = 4

survey_pca_reduced = PCA(n_components = 4,
                           random_state = 508)

survey_pca_reduced.fit(X_scaled_reduced)

# Analyze factor loadings to understand principal components
factor_loadings_df = pd.DataFrame(pd.np.transpose(survey_pca_reduced.components_))

factor_loadings_df = factor_loadings_df.set_index(survey_features_reduced.columns['iPhone':])

#factor_loadings_df.rows = survey_features_reduced.columns

print(factor_loadings_df)

factor_loadings_df.to_excel('mbile_app_factor_loadings.xlsx')

"""Created a dataframe containing the factor loadings to be further analyzed to 
understand the principal components better. Exporting to excel to manipulate the
data easier for visualization."""

#Analyze factor strengths per survey
X_pca_reduced = survey_pca_reduced.transform(X_scaled_reduced)
X_pca_df = pd.DataFrame(X_pca_reduced)

#Rename your principal components and reattach demographic information

X_pca_df.columns = ['App_Lovers', 'Knowledge_Seekers','Entertainers', 
                    'Bargain_Techies']


final_pca_df = pd.concat([mobile_app_df.loc[ : , ['Age', 'Education',
                                                  'Marital_Status',
                                                  'Household_Income',
                                                  'Gender',
                                                  'Race']], 
                                                    X_pca_df], axis = 1)

############################################################################
#KMeans Clustering
############################################################################
# Define the Number of Clusters

survey_k = KMeans(n_clusters = 4,
                      random_state = 508)

survey_k.fit(X_scaled_reduced)

survey_kmeans_clusters = pd.DataFrame({'cluster': survey_k.labels_})

print(survey_kmeans_clusters.iloc[: , 0].value_counts())

"""Split the data into 4 clusters. The following is the breakdown of the split.
    cluster 0: 367, cluster 1: 364, cluster 2: 586, and cluster 3: 205."""

#Analyze cluster centers

centroids = survey_k.cluster_centers_

centroids_df = pd.DataFrame(centroids)

# Renaming columns
centroids_df.columns = survey_features_reduced.columns

print(centroids_df)

# Sending data to Excel
centroids_df.to_excel('final_survey_k4_centriods.xlsx')

centroids_df.describe().round(4)

#############################################################################
#Analyzing Cluster Memberships
############################################################################
X_scaled_reduced_df = pd.DataFrame(X_scaled_reduced)


X_scaled_reduced_df.columns = survey_features_reduced.columns


clusters_df = pd.concat([survey_kmeans_clusters,
                         X_scaled_reduced_df],
                         axis = 1)

print(clusters_df)

###############################################################################
#PCA and Clustering Combined
###############################################################################
#Take your transformed dataframe
print(X_pca_df.head(n = 5))

print(pd.np.var(X_pca_df))

#Scale to get equal variance
scaler = StandardScaler()

scaler.fit(X_pca_df)

X_pca_clust = scaler.transform(X_pca_df)

X_pca_clust_df = pd.DataFrame(X_pca_clust)

print(pd.np.var(X_pca_clust_df))

X_pca_clust_df.columns = X_pca_df.columns

#Define the number of clusters

survey_k_pca = KMeans(n_clusters = 4,
                         random_state = 508)

survey_k_pca.fit(X_pca_clust_df)


survey_kmeans_pca = pd.DataFrame({'cluster': survey_k_pca.labels_})

print(survey_kmeans_pca.iloc[: , 0].value_counts())

"""In the combined pca and cluster dataset the breakdown of the split is the 
    following: cluster 0: 556, cluster 1: 486, cluster 2: 223, and cluster 3: 287."""

########################
#Analyze cluster centers
########################

centroids_pca = survey_k_pca.cluster_centers_


centroids_pca_df = pd.DataFrame(centroids_pca)


# Rename your principal components
centroids_pca_df.columns = ['App_Lovers', 'Knowledge_Seekers','Entertainers', 
                            'Bargain_Techies']

print(centroids_pca_df)

# Sending data to Excel
centroids_pca_df.to_excel('survey_pca_centriods.xlsx')


########################
# Analyze cluster memberships
########################

clst_pca_df = pd.concat([survey_kmeans_pca,
                         X_pca_clust_df],
                         axis = 1)

print(clst_pca_df)

########################
#Reattach demographic information
########################

final_pca_clust_df = pd.concat([mobile_app_df.loc[ : , ['Age', 'Education',
                                                        'Marital_Status',
                                                        'Household_Income',
                                                        'Gender',
                                                        'Race']],
                                clst_pca_df],
                                axis = 1)


print(final_pca_clust_df.head(n = 5))

########################
#Analyze in more detail 
########################
# Renaming Age
ages = {1 : 'Under_18',
        2 : '18_24',
        3 : '25_29',
        4 : '30_34',
        5 : '35_39',
        6 : '40_44',
        7 : '45_49',
        8 : '50_54',
        9 : '55_59',
        10: '60_64',
        11: '65_over'}


final_pca_clust_df['Age'].replace(ages, inplace = True)

# Renaming Education
education_levels = {1 : 'Some_High_School',
                    2 : 'High_School_Graduate',
                    3 : 'Some College',
                    4 : 'College_Graduate',
                    5 : 'Some_Post_Graduate_Studies',
                    6 : 'Post_Graduate_Degree'}

final_pca_clust_df['Education'].replace(education_levels, inplace = True)

#Renaming Martial Status
marital_status = {1 : 'Married',
                  2 : 'Single',
                  3 : 'Single_With_Partner',
                  4 : 'Separated_Widowed_Divorced'}

final_pca_clust_df['Marital_Status'].replace(marital_status, inplace = True)

#Renaming Household Income

household_income = {1 : 'Under_10,000',
                    2 : '10,000_14,999',
                    3 : '15,000_19,999',
                    4 : '20,000_29,999',
                    5 : '30,000_39,999',
                    6 : '40,000_49,999',
                    7 : '50,000_59,999',
                    8 : '60,000_69,999',
                    9 : '70,000_79,999',
                    10: '80,000_89,999',
                    11: '90,000_99,999',
                    12: '100,000_124,999',
                    13: '125,000_149,999',
                    14: '150,000_over'}


final_pca_clust_df['Household_Income'].replace(household_income, inplace = True)

#Renaming Gender
gender = {1: 'Male',
          2: 'Female'}

final_pca_clust_df['Gender'].replace(gender, inplace = True)

#Renaming Race
race = {1: 'White_Caucasian',
        2: 'Black_African_American',
        3: 'Asian',
        4: 'Native_Hawian',
        5: 'American_Indian',
        6: 'Other_Race'}

final_pca_clust_df['Race'].replace(race, inplace = True)



##############################################################################
# Exploring  - "App Lovers"
##############################################################################

#Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Age',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""No real significance"""

#Education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Education',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""No real significance"""

#Marital Status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Marital_Status',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""No real significance"""

#Household Income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Household_Income',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Income Levels 60K-69K, 40K-49K, and 150K have high outliers.15K - 19K have 
a significant cluster 2 and cluster 0.""" 

#Gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Gender',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""Cluster 2 for both male and female is higher"""

#Race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Race',
            y = 'App_Lovers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""White and Black race have high outliers. Native Hawaian cluster 2 stands out"""

##############################################################################
# Exploring  - "Knowledge Seekers"
##############################################################################

#Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Age',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""Cluster 3 stands out in most ages. Age groups that have some significance
are: Under 18, 18-24, 25-29"""

#Education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Education',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Cluster 3 stands out. Some High School and High School Graduate categories 
stand out."""

#Marital Status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Marital_Status',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()
"""Cluster 3 stands out. Single and Single with partner is the category that 
stands out."""

#Household Income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Household_Income',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()
"""Cluster 3 stands out. 15-19K, 30-39K, 40-49K potential target groups.
50-59K, 80-89K, 150+ all have significant outliers."""

#Gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Gender',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""Cluster 3 stands out. No real significance between gender"""

#Race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Race',
            y = 'Knowledge_Seekers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()
"""Cluster 3 stands out. The race categories that stand out are black, asian, 
and hawian"""

##############################################################################
# Exploring  - "Entertainers"
##############################################################################

#Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Age',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()
"""Cluster 0 and 2 and  age category 45-49 stand out.""" 

#Education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Education',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""No real significance"""

#Marital Status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Marital_Status',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10) 
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Maybe the single category. Other than that no real significance."""

#Household Income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Household_Income',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Clusters 0 and 2 stand out. Under 10K and 20-29K age category."""

#Gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Gender',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""No real significance."""

#Race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Race',
            y = 'Entertainers',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Maybe the white and black races."""

##############################################################################
# Exploring  - "Bargin Techies"
##############################################################################

#Age
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Age',
            y = 'Bargain_Techies',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""Cluster 2 and  30-34, 45-49, 55-59 age categories stand out."""

#Education
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Education',
            y = 'Bargain_Techies',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Cluster 2 and College Graduate and Post Graduate Degree Education categories
stand out."""

#Marital Status
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Marital_Status',
            y = 'Bargain_Techies', 
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Cluster 2 and the married category stand out."""

#Household Income
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Household_Income',
            y = 'Bargain_Techies',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Cluster 2 and 60-79K, 100-150K+ are the age categories that stand out."""

#Gender
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Gender',
            y = 'Bargain_Techies',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.show()

"""Cluster 2 stands out and the female group stands out."""

#Race
fig, ax = plt.subplots(figsize = (8, 4))
sns.boxplot(x = 'Race',
            y = 'Bargain_Techies',
            hue = 'cluster',
            data = final_pca_clust_df)

plt.ylim(-10, 10)
plt.tight_layout()
plt.setp(ax.get_xticklabels(), rotation=90)
plt.show()

"""Cluster 2 stands out and the white race category."""

###########################################################################


