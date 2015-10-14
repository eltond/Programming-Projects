
# coding: utf-8

# In[12]:

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import pylab as py
import numpy as np

source = pd.read_csv('seaflow.csv')
Data = pd.DataFrame(source)

pca1=PCA(3)
X_proj = pca1.fit_transform(Data)
print pca1.explained_variance_ratio_
#Checking for Components ability to describe total volume of Data

pca2=PCA(np.sum(pca1.explained_variance_ratio_[0]+pca1.explained_variance_ratio_[1]))
#The above two components describe over 98% of the Data. Used both, and ignored the third.
X_new = pca2.fit_transform(Data)
print pca2.explained_variance_ratio_

plt.scatter(X_new[:,0], X_new[:,1], c='red')

plt.show()


# In[ ]:



