#!/usr/bin/env python
# coding: utf-8

# # 🔍 **Project Overview:**
# - Using unsupervised clustering to segment customers from a groceries firm's database.
# - **Importance:** Customer segmentation optimizes marketing, products, and service by grouping similar customers.
# 
# # 🎯 **Benefits of Segmentation:**
# - **Personalization:** Tailoring interactions and offerings to specific customer needs.
# - **Enhanced Satisfaction:** Addressing individual needs boosts overall satisfaction and loyalty.
# 
# # 🚀 **Significance of Clustering:**
# - **Effective Understanding:** Helps businesses understand and cater to their customer base.
# - **Decision-Making:** Provides actionable insights for strategic decision-making.
# 
# # 📊 **Visualizing Insights:**
# - Creating a dashboard to summarize key findings and provide a clear overview of customer segments.
# - **Goal:** Provide actionable insights for strategic decision-making.

# # Importing Libraries

# In[121]:


import numpy as np
import pandas as pd
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans


# # Data Validity

# In[122]:


df_main= pd.read_csv(r'C:\Users\HP\Downloads\Marketing Analysis Project\marketing_campaign.csv',delimiter='\t')
print('Number of records:',len(df_main))
df_main.head()


# We Have here 2240 rows\
# &\
# 29 different columns

# In[123]:


df_main.info()


# Summary:\
# Missing Values:\
# Income Column: 24 missing records.\
# Other Columns: No missing values found.\
# Data Types:\
# Dt-Customer Column: Currently stored as an object, not a Date type.\
# Categorical Columns:\
# Certain columns are categorical and will require encoding.

# In[124]:


## Rows with missing values in income will be dropped because its number is pretty small compared to the over all number

df_main= df_main.dropna()
print('New number of records after dropping missing records is:',len(df_main))


# In[125]:


df_main.describe()


# # Feature Engineering

# Features to be created:
# 1. Age out of year of birth column
# 2. From kidhome and teenhome, generate a feature that tells if a customer is a parent or not
# 3. Dt_customer, i can generate a feature that tells how many days each customer has with the company
# 4. Total spent amount by each customer
# 5. column tells if that person live with his significant other or alone out of marital status column
# 6. Total number of purchases across different sources
# 7. Total number of children either teen or kid
# 8. Number of memebers in each consumer's family
# 9. Feature that makes education level reduced to only three simple values
# 10. From summary statitics it was noticed that Z_CostContac & Z_Revenue needed to be removed

# In[126]:


# Feature NO. 3
df_main['Dt_Customer']= pd.to_datetime(df_main['Dt_Customer'])
Newest_date= df_main['Dt_Customer'].max()
df_main['Newest_date']= Newest_date
df_main['Days_with_company']= df_main['Newest_date']-df_main['Dt_Customer']
df_main["Days_with_company"] = pd.to_numeric(df_main["Days_with_company"])

## To Convert nano seconds into days as when doing math between dates it gives Timedelta which converted into nano-seconds
## through pd.to_numeric() method

df_main['Days_with_company']= df_main['Days_with_company']/ (1e9 * 60 * 60 * 24)


# #### Exploring Categorical features before creating new ones (Education & Marital Status)

# In[127]:


# Set the style of the plot
sns.set_style("whitegrid")

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot the first bar chart for 'Education'
sns.countplot(x='Education', data=df_main, ax=ax1, palette='pastel')
ax1.set_title('Education Distribution', fontsize=14)

# Plot the second bar chart for 'Marital_Status'
sns.countplot(x='Marital_Status', data=df_main, ax=ax2, palette='pastel')
ax2.set_title('Marital Status Distribution', fontsize=14)

# Adjust the layout
plt.tight_layout()

# Display the plot
plt.show()


# The distrubions suggest to create new features with smaller number of categories as some categories in each distribution are more dominant while some other categories exists in very small numbers such as (Alone, Absurd, Yolo), (Basic, 2n Cycle)

# In[128]:


# Feature NO. 1
df_main['Age']= 2024- df_main['Year_Birth']
# Feature NO. 2
def is_parent(row):
    if row['Kidhome'] > 0 or row['Teenhome'] > 0:
        return '1'
    else:
        return '0'

# Apply the function to create a new column
df_main['Parent_Status'] = df_main.apply(lambda row: is_parent(row), axis=1)
# Feature NO. 4
df_main['Total_Spent_Am']= df_main['MntFruits']+df_main['MntMeatProducts']+df_main['MntFishProducts']\
+df_main['MntSweetProducts']+df_main['MntGoldProds']+df_main['MntWines']
# Feature NO. 5
marital_dict={'Single':'Alone','Divorced':'Alone','Widow':'Alone','Alone':'Alone','Absurd':'Alone',\
              'YOLO':'Alone','Together':'Partner','Married':'Partner'}
df_main['Marital_Status2']= df_main['Marital_Status'].map(marital_dict)
# Feature NO. 6
df_main['Total_Purchased_Number']= df_main['NumWebPurchases']+df_main['NumCatalogPurchases']+df_main['NumStorePurchases']\
+df_main['NumDealsPurchases']
# Feature NO. 7
df_main['Total_children_num']= df_main['Kidhome']+df_main['Teenhome']
# Feature NO. 8
df_main['Fam_Size']= np.where(df_main.Marital_Status2=='Alone',1+df_main.Total_children_num,2+df_main.Total_children_num)
# Feature NO. 9
edu_dict={'PhD':'Postgraduate','Master':'Postgraduate','Graduation':'Graduate',\
          'Basic':'Undergraduate','2n Cycle':'Undergraduate'}
df_main['Education2']= df_main['Education'].map(edu_dict)
# Finally drop columns that are not important for clustering
to_drop = ["Marital_Status", "Education", "Dt_Customer", "Z_CostContact", "Z_Revenue", "Year_Birth", "ID","Newest_date"]
data = df_main.drop(to_drop, axis=1)


# In[129]:


pd.options.display.float_format = '{:,.2f}'.format
# Describe the DataFrame
print(data.describe())


# ##### Notes from describing the Data:
# 1. Income column seems to have Outliers as 75% is below 69K while the maximum value is 667K
# 2. It also seems that the highest averages of money paid by people are in wine, followed by meat
# 3. Age also suggest that some of the records must be dead as i am calculating the age relevant to 2024 and the data is pretty old. For example, maximum age is 131

# In[130]:


# Set up colors preferences
sns.set(rc={"axes.facecolor": "#D3D3D3", "figure.facecolor": "#D3D3D3"})
color_dict = {"1": "#FF69B4", "0": "#00008B"}

# Set up font properties for bold black text
prop = fm.FontProperties(weight='bold', size=12)
plt.rcParams.update({'axes.labelcolor': 'black'})

# Plotting selected features
To_Plot = ["Income", "Recency", "Days_with_company", "Age", "Total_Spent_Am", "Parent_Status"]
sns.pairplot(data[To_Plot], hue="Parent_Status", palette=color_dict)

# Set text color to black
for ax in plt.gcf().axes:
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.title.set_color('black')
    for t in ax.texts:
        t.set_color('black')
        t.set_fontproperties(prop)

plt.show()


# Here we can conclude that Outliers in Both age and Income are represented in small number of records so I will remove these records

# In[131]:


remove_age= (data['Age']<100)
remove_income= (data['Income']<200000)
data= data.loc[remove_age]
data= data.loc[remove_income]
print('Now we have:',len(data),'Records')


# In[132]:


data = pd.get_dummies(data, columns=['Education2', 'Marital_Status2'])
data['Parent_Status']=data['Parent_Status'].astype(int)

print('All features are numerical now')


# In[133]:


#Creating a copy of data
data_final = data.copy()
# creating a subset of dataframe by dropping the features on deals accepted and promotions
cols_del = ['AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'AcceptedCmp1','AcceptedCmp2', 'Response']
data_final = data_final.drop(cols_del, axis=1)
#Scaling
scaler = StandardScaler()
scaler.fit(data_final)
scaled_data_final = pd.DataFrame(scaler.transform(data_final),columns= data_final.columns )
print("All features are now scaled")


# In[134]:


scaled_data_final.head() ## to be used in modeling


# #### Applying Dimensionality Reduction

# In[135]:


## First I need to look at correlations between Features
# Create a correlation matrix
correlation_matrix = scaled_data_final.corr()

# Set up colors preferences
sns.set(rc={"axes.facecolor": "#D3D3D3", "figure.facecolor": "#D3D3D3"})
colors = ["#FFC0CB", "#808080", "#ADD8E6"]  # Pink, gray, blue colors

# Create a heatmap with annotations
plt.figure(figsize=(22, 20))
sns.heatmap(correlation_matrix, cmap=colors, annot=True, fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()


# In[136]:


#PCA to reduce features to 3
pca = PCA(n_components=3)
pca.fit(scaled_data_final)
PCA_ds = pd.DataFrame(pca.transform(scaled_data_final), columns=(["col1","col2", "col3"]))


# In[137]:


# Quick examination of elbow method to find numbers of clusters to make.
print('Elbow Method to determine the number of clusters:')
Elbow_M = KElbowVisualizer(KMeans(), k=10)
Elbow_M.fit(PCA_ds)
Elbow_M.show()


# It indicates that we Have in our Data Four clusters

# In[138]:


# Define the number of clusters based on the elbow method or silhouette score
n_clusters = 4  # Update with the optimal number of clusters

# Create a KMeans model
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

# Fit the model to your data
cluster_labels = kmeans.fit_predict(PCA_ds)

# Add the cluster labels to your original dataset
PCA_ds['Cluster'] = cluster_labels
data['Cluster']= cluster_labels


# In[139]:


from mpl_toolkits.mplot3d import Axes3D

# Create a 3D scatter plot
fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(121, projection='3d')
scatter = ax.scatter(PCA_ds['col1'], PCA_ds['col2'], PCA_ds['col3'], c=PCA_ds['Cluster'], cmap='viridis')
ax.set_xlabel('col1')
ax.set_ylabel('col2')
ax.set_zlabel('col3')
plt.title('3D Scatter Plot of Clusters')

# Add a colorbar
plt.colorbar(scatter, ax=ax, label='Cluster')

# Bar chart of cluster distribution
cluster_counts = PCA_ds['Cluster'].value_counts().sort_index()
plt.subplot(122)
sns.barplot(x=cluster_counts.index, y=cluster_counts.values, palette='viridis')
plt.xlabel('Cluster')
plt.ylabel('Count')
plt.title('Cluster Distribution')
plt.tight_layout()

plt.show()


# In[148]:


data


# In[145]:


remove_age= (df_main['Age']<100)
remove_income= (df_main['Income']<200000)
df_main= df_main.loc[remove_age]
df_main= df_main.loc[remove_income]
print('Now we have:',len(df_main),'Records')


# In[146]:


df_main['Cluster']= cluster_labels


# In[149]:


df_main.to_excel('Marketing_Segmentation.xlsx',index=False)


# In[ ]:




