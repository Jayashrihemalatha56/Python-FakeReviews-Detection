#!/usr/bin/env python
#python 3.7 32 bit
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#get_ipython().run_line_magic('matplotlib', 'inline')

#Import Cancer data from the Sklearn library
# Dataset can also be found here (http://archive.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+%28diagnostic%29)

#from sklearn.datasets import load_breast_cancer
#cancer = load_breast_cancer()
#from google.colab import files
#uploaded = files.upload()
data = pd.read_csv("reviewssvm.csv")
def strip_non_ascii(string):
    ''' Returns the string without non ASCII characters'''
    stripped = (c for c in string if 0 < ord(c) < 127)
    return ''.join(stripped)


data.head()
cancer=data

data['title']=data['title'].str.replace("€", "")
data['title']=data['title'].str.replace("™", "")
data['title']=data['title'].str.replace("â", "")
data['text']=data['text'].str.replace("€", "")
data['text']=data['text'].str.replace("™", "")
data['text']=data['text'].str.replace("â", "")

data['text']=data['text'].str.replace(",", "")
data['title']=data['title'].str.replace(",", "")


data['category']=data['category'].astype('category')
data['title']=data['title'].astype('category')
data['text']=data['text'].astype('category')
data.replace(np.nan,0)
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna()
data = data.reset_index()
#newsline=strip_non_ascii(i)
# In[2]:


#print(cancer)
#print(cancer.columns)


# As we can see above, not much can be done in the current form of the dataset. We need to view the data in a better format.

# # Let's view the data in a dataframe.

# In[3]:


df = data#pd.DataFrame(np.c_[cancer['title'], cancer['category']], columns = np.append(cancer['category'], ['category']))

df.head()




df.shape


# As we can see,we have 596 rows (Instances) and 31 columns(Features)

# In[5]:


print(df.columns)
#df=df.replace(to_replace=None, value=None, inplace=False, limit=None, regex=False, method='pad')
#df['category']=df['category'].astype('category')
#df['title']=df['title'].astype('category')
cat_columns = df.select_dtypes(['category']).columns
cat_columns
df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
print(df.head())
#exit()
# Above is the name of each columns in our dataframe.
# # The next step is to Visualize our data
# In[6]:

print(df.columns.values)
# Let's plot out just the first 5 variables (features)
#sns.pairplot(df, hue = 'category', vars = ['title',  'category'])


# **Note:**
#
#   1.0 (Orange) = Benign (No Cancer)
#
#   0.0 (Blue) = Malignant (Cancer)

# # How many Benign and Malignant do we have in our dataset?

# In[8]:


df['category'].value_counts()


# As we can see, we have 212 - Malignant, and 357 - Benign

#  Let's visulaize our counts

# In[9]:


sns.countplot(df['category'], label = "Count")


# # Let's check the correlation between our features

# In[10]:


plt.figure(figsize=(20,12))
sns.heatmap(df.corr(), annot=True)




X = df.drop(['category'], axis = 1) # We drop our "target" feature and use all the remaining features in our dataframe to train the model.
X.head()


# In[12]:


y = df['category']
y.head()


# # Create the training and testing data

# Now that we've assigned values to our "X" and "y", the next step is to import the python library that will help us to split our dataset into training and testing data.

# - Training data = Is the subset of our data used to train our model.
# - Testing data =  Is the subset of our data that the model hasn't seen before. This is used to test the performance of our model.

# In[13]:


from sklearn.model_selection import train_test_split


# Let's split our data using 80% for training and the remaining 20% for testing.

# In[14]:

indices =range(len(df))
leng=len(df)
X_train, X_test, y_train, y_test,tr,te = train_test_split(X, y,indices, test_size = 0.2, random_state = 20)


# Let now check the size our training and testing data.

# In[15]:


print ('The size of our training "X" (input features) is', X_train.shape)
print ('\n')
print ('The size of our testing "X" (input features) is', X_test.shape)
print ('\n')
print ('The size of our training "y" (output feature) is', y_train.shape)
print ('\n')
print ('The size of our testing "y" (output features) is', y_test.shape)


# # Import Support Vector Machine (SVM) Model

# In[16]:


from sklearn.svm import SVC


# In[17]:


svc_model = SVC()


# # Now, let's train our SVM model with our "training" dataset.

# In[18]:


svc_model.fit(X_train, y_train)

import pickle

# Save the trained SVM model
with open('prg2SVMFakeReviews.pkl', 'wb') as f:
    pickle.dump(svc_model, f)
# # Let's use our trained model to make a prediction using our testing data

# In[19]:


y_predict = svc_model.predict(X_test)


from sklearn.metrics import classification_report, confusion_matrix,accuracy_score


# In[21]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_fake', 'is_real'],
                         columns=['predicted_fake','predicted_real'])
print(confusion)


# In[22]:


sns.heatmap(confusion, annot=True)


# In[23]:


print(classification_report(y_test, y_predict))




X_train_min = X_train.min()
X_train_min


# In[25]:


X_train_max = X_train.max()
X_train_max


# In[26]:


X_train_range = (X_train_max- X_train_min)
X_train_range


# In[27]:


X_train_scaled = (X_train - X_train_min)/(X_train_range)
X_train_scaled.head()


# # Normalize Training Data

# In[28]:


X_test_min = X_test.min()
X_test_range = (X_test - X_test_min).max()
X_test_scaled = (X_test - X_test_min)/X_test_range


# In[29]:


svc_model = SVC()
svc_model.fit(X_train_scaled, y_train)


# In[30]:


y_predict = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_predict)


# # SVM with Normalized data

# In[31]:


cm = np.array(confusion_matrix(y_test, y_predict, labels=[1,0]))
confusion = pd.DataFrame(cm, index=['is_fake', 'is_real'],
                         columns=['predicted_fake','predicted_real'])
confusion


# In[32]:


sns.heatmap(confusion,annot=True,fmt="d")


# In[33]:
print(classification_report(y_test,y_predict))
for i in range(0,len(y_predict)):
   print(te[i] , ":",  y_predict[i])
ac = accuracy_score(y_test,y_predict)
print('Accuracy:')
print(round(ac,2))
print('Confusion Matrix')
print(confusion_matrix(y_test, y_predict))
cm=confusion_matrix(y_test, y_predict)
# Step 3: Extract TP, FP, TN, FN from Confusion Matrix (for binary classification)
TP = cm[0, 0]  # True Positive: Correctly predicted positive
TN = cm[1, 1]  # True Negative: Incorrectly predicted negative
FP = cm[0, 1]  # False Positive: Incorrectly predicted positive
FN = cm[1, 0]  # False Negative: Correctly predicted negative

# Step 4: Calculate Precision, Recall, and F1 Score using TP, FP, TN, and FN

precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (TP + TN) / (TP + TN + FP + FN)

# Output the accuracy
print(f"Accuracy: {accuracy:.4f}")
# Output the Confusion Matrix and Scores
print(f"Confusion Matrix:\n{cm}")
print(f"True Positive (TP): {TP}")
print(f"False Positive (FP): {FP}")
print(f"True Negative (TN): {TN}")
print(f"False Negative (FN): {FN}")

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1_score:.4f}")

X=cm.flatten(order='C')
values=X
#xaxistitles=['HH','HM','HS','MH','MM','MS','SH','SM','SS']#['True Positive','True Negative','False
xaxistitles=['True Positive','False Positive','True Negative','False Negative']#[ '500-1000','1001-1500','1501-2000','2001-2500','2501-3000','3001-3500']
#for i in range(0,len(values)):
  #xaxistitles.append(str(dominantcloudservers[i]))
#plt.bar(xaxistitles,values)
#plt.scatter(DCCap, df['y'], color=colors, alpha=0.5, edgecolor='k')
#for idx, centroid in enumerate(centroids):
#    plt.scatter(*centroid, color=colmap[idx+1])
#plt.xlim(0, 7)
#plt.ylim(0, 40)
#plt.show()

def autolabel(rects,ax):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%d' % int(height),
                ha='center', va='bottom')

ind = np.arange(4)#9)  # the x locations for the groups
width = 0.35
men_std = (2, 2, 2, 2)#,2,2,2,2,2)
fig, ax = plt.subplots()
rects1 = ax.bar(ind, values, width, color='r', yerr=men_std)
 #women_means = (25, 32, 34, 20, 25,25, 32, 34, 20, 25)
 #women_std = (3, 5, 2, 3, 3,2,2,2,2,2)
 #rects2 = ax.bar(ind + width, women_means, width, color='y', yerr=women_std)
 # add some text for labels, title and axes ticks
ax.set_xlabel('CATEGORY')
ax.set_ylabel('RECORDS COUNT')
ax.set_title('CONFUSION MATRIX VALUES')
ax.set_xticks((ind + width / 2 ) )
ax.set_xticklabels(xaxistitles) #  ('1', '2', '3', '4', '5','6','7','8','9','10'))
 #ax.legend((rects1[0], rects2[0]), ('Men','Women'))
ax.legend(['Value'])
autolabel(rects1,ax)
 #autolabel(rects2)
#plt.savefig('FlaskDeployedApp/static/outputimages/prgsvm.png')
plt.show()
exit()
