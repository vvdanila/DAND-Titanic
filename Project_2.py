
# coding: utf-8
# 
# # DAND Project 2: Investigate a Dataset
# ## Dat Set: Titanic Passengers List
# 
# ### Start Date: 29 November 2016
# ### Submission Date: 30 November 2016
# ### Author: Victor Danila

# ## Questions investigated:
# ### - What factors made people more likely to survive?

# In[1]:

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import io
get_ipython().magic(u'matplotlib inline')


# ### Original Data Frame

# In[2]:

filename = 'titanic-data.csv'
titanic_df = pd.read_csv(filename)
titanic_df.head(n=5)


# ### Checking for Missing Values and Filling in Missing Instances  (Data Wrangling)

# In[3]:

# Show columns with missing values
print 'Columns with missing values:'
print
print titanic_df.isnull().any()
print
# Count missing instances in Age column:
print 'Number of missing instances in Age columns:', sum(titanic_df['Age'].isnull())
print 'Percent of missing instances in Age Column:', sum(1.0 * titanic_df['Age'].isnull())/len(titanic_df.index)*100
print 
# Add an Age column with rounded values:
titanic_df['Age_Rounded'] = titanic_df['Age'].round()
# Set login for filling in missing values:
print 'Age mode grouped by Class and Gender:'
print
ref_table = titanic_df.groupby(['Pclass', 'Sex'], as_index=False)['Age_Rounded'].agg(lambda x:x.value_counts().index[0])
print ref_table


# I will fill in the missing age instances given that I will be looking how age affects survival chances. 

# In[11]:

def fill_age(age):
    if np.isnan(age):
        return 999
    else:
        return age




# In[10]:

for age in titanic_df['Age']:
    print titanic_df['Age'].index.get_loc(age)


# In[88]:

# Function that fills in the missing age values depensing on gender
#def fill_age(age_series):
#


# ### Latest Data Frame (After Data Wrangling)

# In[15]:

#Function that assigns passenger to an age group:
def assign_age_group(age):
    if age is not 'NaN':
        if age <= 9:
            return '0_Children'
        elif 10 <= age <= 19:
            return '1_Adolescents'
        elif 20 <= age <= 45:
            return '2_Adults'
        elif 46 <= age <=60:
            return '3_Middle Age Adults'
        elif 60 < age:
            return '4_Seniors'
        
#Creates new column named 'Age Group':
titanic_df['Age Group'] = titanic_df['Age'].apply(assign_age_group)

#Function that assigns 1 to each passanger, to be used in counting. 
def assign_num_value(value):
    return 1

#Creates new column and insert 1 on each row:
titanic_df['Count'] = titanic_df['PassengerId'].apply(assign_num_value)
titanic_df.head(n=5)


# # Part I - Data Grouped by Class and Gender

# ### Graph  of Survival Ratio by Class and Gender

# In[4]:

survival_ratio_data = titanic_df.groupby(['Pclass', 'Sex',], as_index=False )['Survived'].mean()
deceased = survival_ratio_data['Survived'] * (-1) + 1.00
survival_ratio_data['Deceased'] = deceased
survival_ratio_data


# In[5]:

plot1_data = survival_ratio_data[['Survived', 'Deceased']]

plot1_data.index = ['Class I - Women', 'Class I - Men', 'Class II - Women', 'Class II - Men', 'Class III - Women',
                    'Class III - Men']
plot1_data


# In[6]:

plotClassGenderRatio = plot1_data.plot(kind='bar', stacked=True, title='Titanic Survival Ratio by Class and Gender',
                       legend=True,)


# ### Graph of survival numbers by class and gender

# In[7]:

plot1_1_data_df = titanic_df.groupby(['Pclass', 'Sex',], as_index=False )['Count', 'Survived'].sum()
plot1_1_data_df['Deceased'] = plot1_1_data_df['Count'] - plot1_1_data_df['Survived']
plot1_1_data_df.drop('Pclass', axis=1, inplace=True)
plot1_1_data_df.drop('Sex', axis=1, inplace=True)
plot1_1_data_df.drop('Count', axis=1, inplace=True)
plot1_1_data_df.index = ['Class I - Women', 'Class I - Men', 'Class II - Women', 'Class II - Men', 'Class III - Women',
                    'Class III - Men']
plot1_1_data_df


# In[8]:

plotClassGenderAbs = plot1_1_data_df.plot(kind='bar', stacked=True, title='Titanic Survival Numbers by Class and Gender',
                       legend=True,)


# ### Graph  of age frequency of passengers

# In[9]:

age_hist_data = titanic_df['Age']
plotAgeHist = age_hist_data.plot(kind='hist', title='Titanic Passengers Age Histogram')


# # Part II - Data Grouped by Class and Age Group

# ### Graph of survival ratio by class and age group

# In[10]:

age_group_survival_ratio_df = titanic_df.groupby(['Pclass', 'Age Group',], as_index=False)['Count', 'Survived'].sum()
age_group_survival_ratio_df['Survived Ratio'] = age_group_survival_ratio_df['Survived'] / age_group_survival_ratio_df['Count']
age_group_survival_ratio_df['Deceased Ratio'] = 1 - age_group_survival_ratio_df['Survived Ratio']
age_group_survival_ratio_df['Deceased'] = age_group_survival_ratio_df['Count'] - age_group_survival_ratio_df['Survived']
age_group_survival_ratio_df


# In[11]:

plot3_data_df = age_group_survival_ratio_df[['Survived Ratio', 'Deceased Ratio']]
plot3_data_df.columns = ['Survived', 'Deceased']
plot3_data_df.index = ['Class I - Children', 'Class I - Adolescents', 'Class I - Adults', 'Class I - Middle Age Adults', 'Class I - Seniors',
                       'Class II - Children', 'Class II - Adolescents', 'Class II - Adults', 'Class II - Middle Age Adults', 'Class II - Seniors',
                       'Class III - Children', 'Class III - Adolescents', 'Class III - Adults', 'Class III - Middle Age Adults', 'Class II - Seniors'
                      ]
plot3_data_df


# In[12]:

plotClassAgeRatio = plot3_data_df.plot(kind='bar', stacked=True, title='Titanic Survival Ratio by Class and Age Group',
                       legend=True,)


# ### Graph of survival numbers by class and age group

# In[13]:

plot4_data_df = age_group_survival_ratio_df[['Survived', 'Deceased']]
plot4_data_df.index = ['Class I - Children', 'Class I - Adolescents', 'Class I - Adults', 'Class I - Middle Age Adults', 'Class I - Seniors',
                       'Class II - Children', 'Class II - Adolescents', 'Class II - Adults', 'Class II - Middle Age Adults', 'Class II - Seniors',
                       'Class III - Children', 'Class III - Adolescents', 'Class III - Adults', 'Class III - Middle Age Adults', 'Class II - Seniors'
                      ]
plot4_data_df


# In[14]:

plotClassAgeAbs = plot4_data_df.plot(kind='bar', stacked=True, title='Titanic Survival Numbers by Class and Age Group',
                      legend=True,)


# # Part III - Data Grouped by Number of Siblings

# ### Graph of Survival Ratio by Number of Siblings

# In[15]:

plot4_data = titanic_df.groupby(['SibSp'], as_index=False)['Survived'].mean()
plot4_data


# In[16]:

plotSibSurvRatio = plot4_data.plot(kind='scatter', x='SibSp', y='Survived')


# In[17]:

plot5_data = titanic_df.groupby(['SibSp', 'Sex'], as_index=False)['Survived'].mean()
plot5_data


# # Conclusions

# After analyzing the data it appears that the change of survival depends on the following:
# 1. The gender had a significant impact on the chance of survival. Women were more likely to survive than men indifferent of cabin class or age. 
# 2. The cabin class had a certain impact on survival change but not as clear as gender. The higher the cabin class the higher the chance of survival. 
# 3. The age also has a significant impact on survival. The smaller the age the bigger the survival chance got across all cabin classes. 
# 4. The number of siblings aboard the vessel seems to have a negative impact on the survival chance. The bigger the number of siblings the smaller the chance of survival got. 

# In[ ]:



