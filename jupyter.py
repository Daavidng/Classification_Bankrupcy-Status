#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Basic import libraries
import numpy as np
from numpy import mean, std
import pandas as pd
from time import time

# Graph plotting
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Model selection and evaluation
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report 
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Suppress Warning
import warnings          
warnings.filterwarnings("ignore")


# In[2]:


# Extract .arff file
from scipy.io.arff import loadarff 

dfs = []
dataset_names = ['dataset/1year.arff',
                 'dataset/2year.arff',
                 'dataset/3year.arff',
                 'dataset/4year.arff',
                 'dataset/5year.arff']

for dataframes in dataset_names:
    dfs.append(pd.DataFrame(loadarff(dataframes)[0]))

# Merge datasets    
dataframe = pd.concat(dfs)


# <hr>

# In[3]:


# In case of distortion on df, use df without reloading the dataset again
df = dataframe.copy()


# In[4]:


df.head()


# In[5]:


# Decode bytes datatype
df['class'] = df['class'].str.decode(encoding = 'UTF-8')
df['class'].value_counts()


# In[6]:


# Rows and Columns
df.shape


# In[7]:


# Descriptive statistics
df.describe()


# In[8]:


df.info()


# ## Imputing Missing Value

# In[9]:


# Number of rows with missing values
nonmiss_rows = len(df.dropna())
miss_rows = len(df) - nonmiss_rows
print(f'Missing / Nonmissing \t\t: {miss_rows} / {nonmiss_rows}')

# Percentage of rows with missing values
print(f'Percentage of missingness \t: {miss_rows/len(df)*100} %')


# In[10]:


# The number of missing values in each attributes
#!pip install missingno
import missingno as msno

msno.bar(df, sort="descending", color="dodgerblue")


# In[11]:


# Multivariate feature imputation
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Fit imputation
imp = IterativeImputer(max_iter=10)
imp.fit(df.dropna())

# Transform imputation
column_names = df.columns
df = pd.DataFrame(imp.transform(df))
df.columns = column_names

# Check if there is any NaN
print("Missing values in df ?", df.isnull().values.any())


# ## Exploratory Data Analysis (EDA)

# ### Box Plot
# Box Plot is one of the most commonly used plots. It can be used to show the distribution of the data.

# In[12]:


t0 = time()

# Subplots
fig, axs = plt.subplots(nrows = 16, ncols = 4, figsize = (30,100))

features = df.columns[:-1].tolist()
attrString = features

# Plotting
count = 0

for i in range(16):
    for j in range(4):
        boxPlot = sns.boxplot(x = "class", y = attrString[count], data = df, ax=axs[i, j]) 
        boxPlot.set_title(attrString[count], fontsize = 14)

        count = count + 1

plt.show()

# Time required 
test_time = time() - t0
print("Time required: %.3fs" % test_time)


# From the boxplot, we notice that all the attributes has outliers. We set the threshold for outliers as 3$\sigma$, that is, removed the points which z-score are larger than 3

# In[13]:


from scipy import stats

df = df[(np.abs(stats.zscore(df.iloc[:,:-1])) < 3).all(axis=1)]


# In[14]:


df.shape


# ### Strip plot

# In[15]:


cols = df.columns[:-1]
k = 0

fig, axs = plt.subplots(8,8, figsize = (14,13))

for i in range(8):
    for j in range(8):
        sns.stripplot(x = df['class'], y = df[cols[k]], data = df, jitter = 0.2, ax = axs[i, j])
        k+=1
        
plt.tight_layout()
plt.show()


# ### Heatmap
# When dealing with data attributes it is always important to know, if and how strongly they are correlated, since some models don’t work well with strongly correlated attributes. Apart from that, a strong correlation between two attributes is an indicator that one of them may be superfluous, because it doesn’t carry much additional information. This knowledge can be used to reduce the number of features and thus make computations for training and evaluating a model more feasible.

# In[16]:


#time
t0 = time()

# Compute Correlations
features = df.columns[:-1].tolist()
corr = df[features].corr()

# Heatmap
plt.figure(figsize = (30,30))

# Whole data
# sns.heatmap(corr,annot = True, fmt = '.1f', cmap = 'coolwarm', square = True, linewidths=.5)

# Getting only the lower triangle
corr_mask = np.triu(corr)
corrPlot = sns.heatmap(corr, annot = True, fmt = '.1f', cmap = 'coolwarm', linewidth = .5, square = True, mask = corr_mask)
plt.show()

# Time required for plotting heatmap of dataset containing 64 attributes
test_time = time() - t0
print("Time required: %.3fs" % test_time)


# <hr>

# ## Train Test Split

# In[17]:


# Define dataset
X = df.iloc[:,:-1]  
y = df['class']

## Train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# In[18]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Standard scaler
sc = StandardScaler()

# MinMax Scaler
msc = MinMaxScaler()


# ## Unsupervised Learning

# ### Principal Component Analysis (PCA)

# In[19]:


# import related library
from sklearn.decomposition import PCA


# In[20]:


# Create PCA object
pca = PCA()

# Scalling data using Standard Scaler
X_scaled = sc.fit_transform(X)

# Fitting and transforming data
pca.fit(X_scaled)
pca_data = pca.transform(X_scaled)

# Formatting data
per_var = np.round(pca.explained_variance_ratio_,decimals=4)
labels = ['']+['PC' + str(x) for x in range (1, len(per_var)+1)]
per_var = np.insert(per_var, 0, 0)
cum_per_var = np.cumsum(per_var)

# put pca_data into a dataframe for better formatting
pca_df = pd.DataFrame(labels, columns=['PC'])
pv_df = pd.DataFrame(per_var, columns=['Percentage Variance'])
cv_df = pd.DataFrame(cum_per_var, columns=['Cummulative Variance'])
df_pca = pd.concat([pca_df, pv_df, cv_df], axis=1)
df_pca.head()


# In[21]:


# Number of principal components for describing 90% of the variance

# Change the threshold to set desired percentage of explained veriance
threshold = 0.91
df_pca[df_pca['Cummulative Variance']<threshold].count()


# In[22]:


# Plotting of PCA analysis

# Bar
fig, ax = plt.subplots(figsize=(20,18))
pca_plt = ax.bar(x=range(1,len(per_var)+1),height=per_var,tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel("Principal Component")
plt.title('Scree Plot')
for p in pca_plt:
    height = p.get_height()
    ax.annotate('{}'.format(height),
      xy=(p.get_x() + p.get_width() / 2, height),
      xytext=(0, 3), # 3 points vertical offset
      textcoords="offset points",
      ha='center', va='bottom')

# Cummulative line
#plt.scatter(x='PC', y='Cummulative Variance',data=df_pca)
#plt.plot(pca_df['PC'],pca_df['Cummulative Variance'])

plt.show()


# ### KMeans

# In[23]:


# Import libraries
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# In[24]:


# Code adapt and modified from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# For visualising results of silhouette scores

def plot_silhouette_score(k, X):
    fig, ax = plt.subplots(1, 1)
    ax.set_xlim([-0.5, 1])
    ax.set_ylim([0, len(X) + (k + 1) * 10])

    clusterer = KMeans(k)
    cluster_labels = clusterer.fit_predict(X)

    silhouette_avg = silhouette_score(X, cluster_labels)
    print( "The average silhouette_score is :", silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(k):
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax.fill_betweenx( np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax.set_title("Silhouette plot")
    ax.set_xlabel("The silhouette coefficient values")
    ax.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax.axvline(x = silhouette_avg, color = "red", linestyle = "--")

    ax.set_yticks([])  # Clear the yaxis labels / ticks
    ax.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])


# First, we try to perform K-means clustring on the original X_scaled data

# In[25]:


inertia_list_ori = []

for clust in range(2,21):
    km = KMeans(n_clusters = clust)
    km.fit(X_scaled)
    inertia_list_ori.append(km.inertia_) 


# In[26]:


# Plot figure
plt.figure(figsize = (8,5))

plt.plot(range(2,21), inertia_list_ori, marker = 'o')
plt.title('Inertia of KMeans with original X_scaled data')
plt.xlabel('Cluster')
plt.ylabel('Inertia')

plt.show()


# From figure above, we can see that there is no elbow.<Br/>
# Hence, based on the result from PCA, we reduce the dimension of features to 27, which described more than 90% of the variance to determined if there exist any cluster in the data

# In[27]:


# Fit X_scaled data to PCA with 27 components
pca = PCA(n_components = 27)
pca.fit(X_scaled)
pca_27 = pca.transform(X_scaled)


# In[28]:


inertia_list_pca_27 = []

for clust in range(2,21):
    km = KMeans(n_clusters = clust)
    km.fit(pca_27)
    inertia_list_pca_27.append(km.inertia_) 


# In[29]:


# Plot figure
plt.figure(figsize = (8,5))

plt.plot(range(2,21), inertia_list_pca_27, marker = 'o')
plt.title('Inertia of KMeans with PCA data')
plt.xlabel('Cluster')
plt.ylabel('Inertia')

plt.show()


# Converging of K-means still not effective eventhough the dimension already reduced to 27<Br/>
# Further reduce the dimension to 3 which describing more than 30% variance in the data

# In[30]:


# Fit X_scaled data to PCA with 3 components
pca = PCA(n_components = 3)
pca.fit(X_scaled)
pca_3 = pca.transform(X_scaled)


# In[31]:


inertia_list_pca_3 = []

for clust in range(2,21):
    km = KMeans(n_clusters = clust)
    km.fit(pca_3)
    inertia_list_pca_3.append(km.inertia_) 


# In[32]:


# Plot figure
plt.figure(figsize = (8,5))

plt.plot(range(2,21), inertia_list_pca_3, marker = 'o')
plt.title('Inertia of KMeans with PCA data')
plt.xlabel('Cluster')
plt.ylabel('Inertia')

plt.show()


# The elbow occur at K = 5 when dimension reduced to 3<Br/>
# Use silhouette score to check the goodness of the cluster

# In[33]:


# Visualise the silhouette score with k = 5
# This cell takes longer time to run
plot_silhouette_score(5, pca_3)


# There are maximum silhouette sample that below the average silhouette score,<Br/>
# We find the silhouette score for k = 4 and k = 6 to see if there any improvement

# In[34]:


# Visualise the silhouette score with k = 4
plot_silhouette_score(4, pca_3)


# In[35]:


# Visualise the silhouette score with k = 6
plot_silhouette_score(6, pca_3)


# We have all the sample silhouette score above average when number of cluster is 4 but the average silhouette score decreased, whereas changing number of cluster to 6 still exist cluster with sample silhouette score below average.<Br/><Br/>
# Hence, we select the number of cluster to be 4

# In[36]:


# Kmeans with 4 clusters
kmeans = KMeans(n_clusters = 4)
kmeans.fit(pca_3)


# In[37]:


# Visualise result in 2D
plt.figure(figsize = (15,5))

plt.subplot(1,3,1)
plt.scatter(pca_3[:,0], pca_3[:,1], c = kmeans.labels_, alpha = 0.8)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker = 'x', c = 'r', s = 100)
plt.xlabel('PC1')
plt.ylabel('PC2')

plt.subplot(1,3,2)
plt.scatter(pca_3[:,0] ,pca_3[:,2], c = kmeans.labels_, alpha = 0.8)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,2], marker = 'x', c = 'r', s = 100)
plt.xlabel('PC1')
plt.ylabel('PC3')

plt.subplot(1,3,3)
plt.scatter(pca_3[:,1], pca_3[:,2], c = kmeans.labels_, alpha = 0.8)
plt.scatter(kmeans.cluster_centers_[:,1], kmeans.cluster_centers_[:,2], marker = 'x', c = 'r', s = 100)
plt.xlabel('PC2')
plt.ylabel('PC3')

plt.tight_layout()
plt.show()


# In[38]:


# Visualise result in 3D
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
xx = pca_3[:,0]
yy = pca_3[:,1]
zz = pca_3[:,2]

c1 = kmeans.cluster_centers_[:,0]
c2 = kmeans.cluster_centers_[:,1]
c3 = kmeans.cluster_centers_[:,2]

ax.scatter(xx, yy, zz, c = kmeans.labels_, s = 40, alpha = 0.8)
ax.scatter(c1, c2, c3, c = 'red', s = 80, marker = 'x')

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.show()


# ## Supervised Learning

# ### Logistic Regression

# In[39]:


# scale data
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[40]:


from sklearn.linear_model import LogisticRegression


# In[41]:


# train data
logreg = LogisticRegression(multi_class='ovr') # using default parameter
logreg.fit(X_train_scaled, y_train)

# predicted output
y_pred = logreg.predict(X_test_scaled)


# In[42]:


# Model evaluation
plot_confusion_matrix(logreg, X_test_scaled, y_test, cmap='binary')
print("Accuracy            : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# In[43]:


y_pred_logreg_prob = logreg.predict_proba(X_test_scaled)[:,1] # Positive class = 1

fpr, tpr, threshold = roc_curve(y_test, y_pred_logreg_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

logreg_auc = roc_auc_score(y_test, y_pred_logreg_prob)
print('ROC AUC score = %.3f' % logreg_auc)


# Hyperparameters control the overfitting or underfitting of the model.  
# We use Grid Search to search optimal values for hyperparameters.   

# In[44]:


lr = LogisticRegression()

#tuning weight for minority class then weight for majority class will be 1-weight of minority class
#Setting the range for class weights
weights = np.linspace(0.0,0.99,10)

#specifying all hyperparameters with possible values
param= {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], "class_weight":[{0:x ,1:1.0 -x} for x in weights]}

# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True)

#Gridsearch for hyperparam tuning
model= GridSearchCV(estimator= lr, param_grid=param, scoring="f1", cv=folds, return_train_score=True)

#train model to learn relationships between x and y
model.fit(X_train_scaled, y_train)


# In[45]:


# print best hyperparameters
print("Best F1 score: ", model.best_score_)
print("Best hyperparameters: ", model.best_params_)


# After fitting the model, extract the best fit values for all specified hyperparameters.  
# Then build Logistic Regression model using the above values by tuning Hyperparameters.  

# In[46]:


#Building Model again with best params
logreg2 = LogisticRegression(class_weight={0:0.11,1:0.89}, C=10, penalty="l2")
logreg2.fit(X_train_scaled, y_train)


# In[47]:


#predict labels on test dataset
y_pred2 = logreg2.predict(X_test_scaled)

# confusion matrix
plot_confusion_matrix(logreg2, X_test_scaled, y_test, cmap='binary')
print(" Accuracy    : ", accuracy_score(y_test, y_pred2))
print(" Sensitivity : ", recall_score(y_test, y_pred2))
print(" Precision   : ", precision_score(y_test, y_pred2))

print("\n", classification_report(y_test, y_pred2))


# In[48]:


# roc & auc

# predict probabilities on Test and take probability for class 1([:1])
y_pred_logreg2_prob = logreg2.predict_proba(X_test_scaled)[:, 1]

fpr, tpr, threshold = roc_curve(y_test, y_pred_logreg2_prob)
plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

logreg_auc = roc_auc_score(y_test, y_pred_logreg2_prob)
print('ROC AUC score = %.3f' % logreg_auc)


# ### Naive Bayes

# Naive Bayes is affected by imbalanced data. Since our data is heavily imbalance, before proceeding forward, we will have to resample them. Before this, I am using GaussianNB (with the technique of undersampling), however, there is also an advanced Naive Bayes called ComplementNB which was specifically designed to deal with imbalanced data.

# In[49]:


from sklearn.naive_bayes import ComplementNB

nbModel = ComplementNB()  # No parameter is tuned (using default parameter)


# Particularly, we have to use MinMaxScaler instead of StandardScaler for Complement Naive Bayes classifier.

# In[50]:


X_train_scaled = msc.fit_transform(X_train)
X_test_scaled = msc.transform(X_test)


# After that, we will fit the model with the scaled train data (X_train_scaled, y_train) and get the predicted values for X_test_scaled.

# In[51]:


## Train model
nbModel.fit(X_train_scaled, y_train)

## Test model
y_pred_nb = nbModel.predict(X_test_scaled)
print('y_pred_nb value  : ', y_pred_nb)

# Original y_test values to be compared
print('\ny_test value \t : ', y_test.values)


# Then, we plot the Confusion Matrix of the predicted values (y_pred_nb) and the orginal values (y_test) and the precise Accuracy, Sensitivity, and Precision of the model in predicting the values.

# In[52]:


## Plot the confusion matrix
confusionMatrix = confusion_matrix(y_test, y_pred_nb)

disp = ConfusionMatrixDisplay(confusion_matrix=confusionMatrix)
disp.plot()
plt.show()


# In[53]:


## Find the accuracy, sensitivity and precision

print("Accuracy \t =  %.5f" % accuracy_score(y_test, y_pred_nb))
print("Sensitivity \t =  %.5f" % recall_score(y_test, y_pred_nb))
print("Precision \t =  %.5f" % precision_score(y_test, y_pred_nb))


# The accuracy we obtain here is considered as acceptable.

# We then obtain the Classification Report, which compares between the predicted values (y_pred_nb) with the orginal values (y_test)

# In[54]:


## Classification Report
report = classification_report(y_test, y_pred_nb)
print(report)


# Here, we evaluate the model using the trained data we have by cross-validating.

# In[55]:


# Define model evaluation method
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)

# Evaluate model
scores = cross_val_score(nbModel, X_train_scaled, y_train, scoring = 'accuracy', cv = cv, n_jobs = -1)

# Summarize result
print('Mean Accuracy = %.5f (%.5f)' % (mean(scores), std(scores)))


# By repeating three times, we obtain an fairly high accuracy using 10-fold.

# Afterwards, we will plot out the ROC curve and observe the pattern.

# In[56]:


# Obtain the probability of y_pred_nb
y_pred_nb_prob = nbModel.predict_proba(X_test_scaled)[:,1]
print(y_pred_nb_prob)

# Find the paramters
fpr, tpr, threshold = roc_curve(y_test, y_pred_nb_prob)

# Plot ROC-curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()

# Evaluation on training using ROC AUC
nb_auc = roc_auc_score(y_test, y_pred_nb_prob)
print('ROC AUC score = %.5f' % nb_auc)


# By look at the graph and the ROC AUC score, we can see that the model has just slightly good ability of predicting class 1 (where the company will face bankrupt) as 1 correctly.

# Next, we will try to perform Hyperparameter Tuning. In Naive Bayes (ComplementNB), we have only one parameter to be tuned (alpha for Laplace smoothing).

# First, we will try to use GridSearchCV and look at the performance.
# <Br/>(Warning: Long process time, around 1min - 2mins)

# In[57]:


## Hyperparameter Tuning

# Will be used later to see the time used for GridSearchCV
t0 = time()

# Parameters
param_grid = {'alpha': np.arange(1.0e-10, 1 + 0.01, 0.01)}

# GridSearchCV method
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)
search = GridSearchCV(nbModel, param_grid, scoring = 'accuracy', cv = cv)
results = search.fit(X_train_scaled, y_train)

print("Mean Accuracy \t =  %.5f" % results.best_score_)
print("Config \t :  %s" % results.best_params_)

# Test time of GridSearchCV
test_time = time() - t0
print("\nGridSearchCV Test Time \t :  %.3fs" % test_time)


# GirdSearchCV seems to take a very long time as we are using cv = RepeatedStratifiedKFold function (using cv = 10 will take around 30 seconds to test), hence we will switch the method and use RandomizedSearchCV instead.

# We then try RandomizedSearchCV.

# In[58]:


## Hyperparameter Tuning

# Will be used later to see the time used for RandomizedSearchCV
t0 = time()

# Parameters
param_grid = {'alpha': np.arange(1.0e-10, 1 + 0.01, 0.01)}

# RandomizedSearchCV method
cv = RepeatedStratifiedKFold(n_splits = 10, n_repeats = 3)
search = RandomizedSearchCV(nbModel, param_grid, scoring = 'accuracy', cv = cv)
results = search.fit(X_train_scaled, y_train)

print("Mean Accuracy \t =  %.5f" % results.best_score_)
print("Config \t :  %s" % results.best_params_)

# Test time of RandomizedSearchCV
test_time = time() - t0
print("\nRandomziedSearchCV Test Time \t :  %.3fs" % test_time)


# We can clearly see that RandomizedSearchCV has faster processing time than GridSearchCV. RandomizedSearchCV is normally preferred when we are tuning a large size of hyperparameters since it has faster processing time. However, RandomizedSearchCV is more computationally expensive than GridSearchCV.
# 
# One thing to note that, here we are set the range of alpha to be [1.0e-10, 1] and not any value higher. This is because using higher alpha values will push the likelihood towards a value of 0.5, i.e., the probability of a word equal to 0.5 for both the positive and negative reviews. Since we are not getting much information from that, it is not preferable. Therefore, it is preferred to use alpha = 1.0.
# 
# Here, we will adapt the result of GridSearchCV where the suggested value for Laplace Smoothing is alpha = 1.0. Since by default (where we don't include any value in the hyperparameter) the values of alpha is 1.0, we don't have to perform any modifications. 

# In[59]:


## Conclusion
print("By performing Naive Bayes predictive modelling on the dataset, we obtain:\n")

print("Accuracy \t =  %.5f" % accuracy_score(y_test, y_pred_nb))
print("Sensitivity \t =  %.5f" % recall_score(y_test, y_pred_nb))
print("Precision \t =  %.5f" % precision_score(y_test, y_pred_nb))
print("ROC AUC score \t =  %.5f" % nb_auc)


# ### Linear Discriminant Analysis (LDA)

# In[60]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[61]:


lda = LDA(n_components=1)

X_train_scaled = lda.fit_transform(X_train, y_train)
X_test_scaled = lda.transform(X_test)


# In[62]:


## Grid search for LDA

# define model evaluation method
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3,)

# define grid
grid = dict()
grid['solver'] = ['svd', 'lsqr', 'eigen']

# define search
search = GridSearchCV(lda, grid, scoring='accuracy', cv=cv, n_jobs=-1)

# perform the search
results = search.fit(X_train_scaled, y_train)

# summarize
print('Config: %s' % results.best_params_)
print('Accuracy: %s' % results.best_score_)


# In[63]:


# evaluate model
# lda default solver is svd which same as the result obtained from GridSearchCV

#fit model
lda.fit(X_train_scaled,y_train)

# evaluate model
scores = cross_val_score(lda, X_train_scaled, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

# summarize result
print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[64]:


# Evaluate test set
y_pred = lda.predict(X_test_scaled)


print("LDA - Bankruptcy \n", confusion_matrix(y_test, y_pred))

# confusion matrix
plot_confusion_matrix(lda, X_test_scaled, y_test, cmap='binary')
print("Accuracy            : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# In[65]:


#ROC Curve
pred_prob = lda.predict_proba(X_test_scaled)[:,1]
fpr, tpr, thresh= roc_curve(y_test, pred_prob)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr,tpr)
plt.title('LDA ROC curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.show()

lda_auc = roc_auc_score(y_test, pred_prob)
print('ROC AUC score = %.3f' % lda_auc)


# ### Desicion Tree

# In[66]:


# scale data
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)


# In[67]:


from sklearn.tree import DecisionTreeClassifier 


# In[68]:


tree = DecisionTreeClassifier()
tree.fit(X_train_scaled, y_train)
y_pred = tree.predict(X_test_scaled)

# result
print("Decision Tree(Default hyperparameter) - Bankruptcy \n", confusion_matrix(y_test, y_pred))

# confusion matrix
plot_confusion_matrix(tree, X_test_scaled, y_test, cmap='binary')
print("\nAccuracy          : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# Apply pre-pruning method to decision tree<Br/>
# Use max_depth to stop the tree from growing into the full tree

# In[69]:


dtc = DecisionTreeClassifier()

# Use GridSearchCV to find the best max_depth
from sklearn.model_selection import GridSearchCV

parameters = {"max_depth": np.arange(2,10)}

grid_tree = GridSearchCV(estimator=dtc, param_grid = parameters, cv = 5)
grid_tree.fit(X_train_scaled, y_train)

print(" Results from Grid Search " )
print("\n The best score across ALL searched params:\n", grid_tree.best_score_)
print("\n The best parameters across ALL searched params:\n", grid_tree.best_params_)


# In[70]:


from sklearn.tree import DecisionTreeClassifier as dtc


# Decision tree with max_depth 3 for visualization

# In[71]:


# tree with max_depth 3 for visualization
tree_depth_3 = dtc(criterion = 'entropy', max_depth = 3)
tree_depth_3.fit(X_train_scaled, y_train)
y_pred = tree_depth_3.predict(X_test_scaled)

# result
print("Decision Tree - Bankruptcy \n", confusion_matrix(y_test, y_pred))

# confusion matrix
plot_confusion_matrix(tree_depth_3, X_test_scaled, y_test, cmap='binary')
print("\nAccuracy            : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# In[72]:


from sklearn.tree import plot_tree

feature = df.columns[:64]
target = df['class'].unique().tolist()

plt.figure(figsize = (23,10))
plot_tree(tree_depth_3, 
          feature_names = feature, 
          fontsize = 16, 
          filled = True)

plt.show()


# Decision Tree with best max_depth

# In[73]:


tree1 = dtc(criterion = 'entropy', max_depth = 6)
tree1.fit(X_train_scaled, y_train)
y_pred = tree1.predict(X_test_scaled)

# result
print("Decision Tree - Bankruptcy \n", confusion_matrix(y_test, y_pred))

# confusion matrix
plot_confusion_matrix(tree1, X_test_scaled, y_test, cmap='binary')
print("\nAccuracy            : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# In[74]:


# flow chart - entropy
feature = df.columns[:64]
target = df['class'].unique().tolist()

plt.figure(figsize = (80,20))
plot_tree(tree1, 
          feature_names = feature, 
          fontsize = 8, 
          filled = True)

plt.show()


# In[75]:


tree2 = dtc(criterion='gini', max_depth=6)
tree2.fit(X_train_scaled, y_train)
y_pred = tree2.predict(X_test_scaled)

# result
print("Decision Tree - Bankruptcy \n", confusion_matrix(y_test, y_pred))

# confusion matrix
plot_confusion_matrix(tree2, X_test_scaled, y_test, cmap='binary')
print("\nAccuracy          : ", accuracy_score(y_test, y_pred))
print("Sensitivity / Recall: ", recall_score(y_test, y_pred))
print("Precision / PPV     : ", precision_score(y_test, y_pred))

print("\n", classification_report(y_test, y_pred))


# In[76]:


# flow chart - gini
plt.figure(figsize = (80,20))
plot_tree(tree2, 
          feature_names = feature, 
          fontsize = 8, 
          filled = True)
plt.show()

