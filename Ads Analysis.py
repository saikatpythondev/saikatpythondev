#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import power_transform,StandardScaler


# In[2]:


df=pd.read_csv(r"F:\download from c to e\KAG_conversion_data.csv")


# In[3]:


df.head(20)


# In[4]:


df.gender.value_counts()


# In[5]:


####null value handelling


# In[6]:


df[df.isna()].head()


# In[7]:


df=df.dropna(axis=0,inplace=False)


# In[8]:


df.describe()


# In[9]:


df.Clicks.unique()


# In[10]:


df.Clicks.median()


# In[11]:


df.Clicks.mean()


# In[12]:


sns.histplot(df.Clicks)


# In[13]:


sns.boxplot(df.Clicks)


# In[ ]:





# In[14]:


sns.countplot(df.gender)


# In[15]:


#### the spent of time in the ads


# In[16]:


df.Spent.describe()


# In[17]:


sns.histplot(df.Spent,kde=True)


# In[18]:


sns.boxplot(data=df,x="gender",y="Spent")


# In[19]:


import scipy.stats as stat


# In[ ]:





# In[20]:


####the histogram looks like log normal distribution so to make it nearly Normal dist we have to do log transform


# In[21]:


sm.qqplot(np.log(df2["Spent"]))
sm.qqplot((df2["Spent"]))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


####outlier detection using Z_Score


# In[22]:


x=df.groupby(["gender"])["Spent"].mean()


# In[23]:


y=df.groupby(["gender"])["Spent"].median()


# In[24]:


print(x)
print(y)


# In[25]:


X=x.keys()


# In[26]:


X


# In[27]:


plt.pie(x,labels=X,autopct="%1.1f%%")
plt.title("mean time")
plt.show()
plt.pie(y,labels=X,autopct="%1.1f%%")
plt.title("median time")
plt.show()


# In[28]:


mu=df.Spent.mean()
std=df.Spent.std()


# In[29]:


z=(df.Spent-mu)/std


# In[30]:


outlier=z[z>=3]


# In[31]:


outlier.count()


# In[32]:


df2=df.drop(["xyz_campaign_id","fb_campaign_id","ad_id"],axis=1)
df2.head()


# In[33]:


from sklearn.preprocessing import LabelEncoder


# In[34]:


le=LabelEncoder()
sex=le.fit_transform(df.gender)


# In[35]:


sex=pd.DataFrame(sex,columns=["sex"])


# In[36]:


####male=1,female=0


# In[37]:


df2=pd.concat([df2,sex],axis=1)


# In[38]:


df2.age.value_counts()


# In[39]:


df2.age.dtype


# In[40]:


df5=df2.age.apply(str)


# In[41]:


df5=le.fit_transform(df5)


# In[42]:


#### age
#30-34=0
#45-49=1
#35-39 =2
#40-44=3


# In[43]:


Age=pd.DataFrame(df5,columns=["Age"])


# In[44]:


df2=pd.concat([df2,Age],axis=1)


# In[45]:


df2


# In[46]:


sns.heatmap(df2.corr())


# In[47]:


df2.corr()


# In[48]:


df2.corr().sum()


# In[49]:


#### from correlation map we see that the colum "Impressions" is highly correlate with other colum so this is dependent feature


# In[61]:


df2.cov()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[50]:


import plotly.express as pl


# In[51]:


fig=pl.scatter_3d(df2,x=df2.Clicks,y=df2.Spent,z=df2.Total_Conversion,color=df2.Impressions)


# In[52]:


fig.show()


# In[116]:


df2.Spent.describe()


# In[117]:


IQR=60.025000-1.480000


# In[118]:


60.025000+(1.5*IQR)


# In[119]:


df7=df2[df2.Spent>=147.8425]
df7.Spent


# In[120]:


df7.count()


# In[121]:


df2.drop([])


# In[ ]:





# In[ ]:





# In[ ]:





# In[122]:


i=df2.drop(["age","gender","Impressions"],axis=1)


# In[123]:


d=df2["Impressions"]


# In[140]:


from sklearn.preprocessing import StandardScaler


# In[141]:


i11=StandardScaler().fit_transform(i)


# In[142]:





# In[124]:


from sklearn.decomposition import PCA


# In[157]:


pca=PCA(n_components=4)
i1=pca.fit_transform(i)


# In[158]:


i1


# In[159]:


from sklearn.linear_model import LinearRegression


# In[160]:


lr=LinearRegression()


# In[161]:


from sklearn.model_selection import cross_val_score


# In[163]:


cross_val_score(lr,i1,d,cv=5)
    


# In[164]:





# In[167]:


from sklearn.model_selection import train_test_split


# In[168]:


x_train,x_test,y_train,y_test=train_test_split(i1,d,test_size=.2)


# In[169]:


lr.fit(x_train,y_train)


# In[170]:


lr.score(x_train,y_train)


# In[171]:


lr.score(x_test,y_test)


# In[ ]:




