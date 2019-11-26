#!/usr/bin/env python
# coding: utf-8

# In[60]:


# Standard and OS :
import json
import glob
import os
import importlib
import multiprocessing as mp

# Image processing :
import skimage
import cv2 as cv

# Numeric :
import numpy as np
import pandas as pd

# Visualisation :
import matplotlib.pyplot as plt
import seaborn as sns


# In[61]:


import mfilt_funcs as mfs
importlib.reload(mfs)
import mfilt_funcs as mfs

import utils
importlib.reload(utils)
import utils


# In[62]:


lmap = lambda x, y: list(map(x, y))
lfilter = lambda x, y: list(filter(x, y))
imread = lambda x: cv.imread(x, 0)


# In[71]:


plt.style.use('seaborn-deep')
plt.rcParams['figure.figsize'] = (12, 8)


# In[64]:


ls images/


# In[65]:


cwd  = os.path.abspath('.')
path = os.path.join(cwd, 'images')
pattern = os.path.join(path, '*flujo.png')
files = glob.glob(pattern)
files


# Todas nuestras imágenes de interés contienen la cadena de caracteres 'flujo.png'.

# In[66]:


#pool = mp.Pool()


# In[67]:


mangueras = {
    f"{nombre}": imread(file) for file, nombre in zip(files, lmap(lambda x: os.path.split(x)[-1], files)) 
}
mangueras


# 

# In[58]:


for nombre in mangueras.keys():
    plt.figure()
    plt.imshow(mangueras[nombre], cmap="gray")
    plt.title(nombre)


# In[72]:


for nombre in mangueras.keys():
    #plt.figure()
    sns.distplot(mangueras[nombre].flatten(), label=nombre)
    plt.title(nombre)


# In[33]:


help(pool.map)


# In[ ]:




