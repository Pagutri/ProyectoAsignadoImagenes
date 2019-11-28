#!/usr/bin/env python
# coding: utf-8

# In[109]:


# Type annotations :
from typing import Tuple, List, Optional, NoReturn, Callable, Any

# Standard and OS :
import copy
import json
import glob
import os
import importlib # Required to reload a module
                 # because the Jupyter Kernel
                 # won't  really reimport by itself.
import multiprocessing as mp

# Image processing :
import cv2 as cv
import skimage
from skimage.feature import canny
from skimage.util.dtype import dtype_range
from skimage.util import img_as_ubyte
from skimage import exposure
from skimage.morphology import disk
from skimage.filters import rank
from skimage.measure import label, regionprops

# Numeric :
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

# Visualisation :
import matplotlib.pyplot as plt
import seaborn as sns

# Machine-Learning :
from sklearn.cluster import KMeans

# Functional programing tools : 
from functools import partial, reduce
from itertools import chain


# In[2]:


import mfilt_funcs as mfs
importlib.reload(mfs)
import mfilt_funcs as mfs

import utils
importlib.reload(utils)
import utils


# In[3]:


lmap = lambda x, y: list(map(x, y))
lfilter = lambda x, y: list(filter(x, y))
imread = lambda x: cv.imread(x, 0)


# In[4]:


plt.style.use('seaborn-deep')
plt.rcParams['figure.figsize'] = (12, 8)


# In[155]:



def auto_segment(
    img: np.ndarray, 
    groups: int = 2,
    nonzero: bool = False,
    verbose: bool = False, 
    save_file: Optional[str] = None,
    figsize: Optional[Tuple[int]] = (12, 8)
) -> np.ndarray:
    """
        Segment (by thresholding)
    """
    
    assert type(groups) is int, f"type(groups) == '{type(groups)}', it should be int."
    
    #Create the destination image from the image passed to the function, casting it when needed.
    _floats = [np.float, np.float16, np.float32, np.float64, np.float128]
    if img.dtype in _floats:
        dst: np.ndarray = copy.deepcopy(img)
    else:
        dst: np.ndarray = copy.deepcopy(np.float64(img) / 255)
    
    # We perform K-Means clustering analysis :
    _intensities = img.flatten()
    _show_intensities = _intensities.copy()
    if nonzero:
        _intensities = _intensities[_intensities.nonzero()]
    _kmeans = KMeans(n_clusters=groups, random_state=0, verbose=verbose).fit(_intensities.reshape(-1, 1))
    _centers = pd.core.frame.DataFrame({
        "means": chain.from_iterable(_kmeans.cluster_centers_)
    })
    _centers = _centers.sort_values(by=['means'])
    
    # We obtain our threshold values as pairwise means between cluster centers.
    _centers['k'] = _centers.rolling(2).mean()
    
    # Create the values that will fill the image, according to the thresholds.
    _fill_vals = np.linspace(0, 1, groups, dtype=np.float64)
    
    # Fill the image with trheshold values.
    ks = [0] + _centers['k'].dropna().tolist()
    for j in range(len(ks) - 1):
        _mask = np.nonzero( (img > ks[j]) & (img < ks[j+1]) )
        dst[ _mask ] = _fill_vals[j]
    _mask = np.nonzero( img > ks[-1] )
    dst[ _mask ] = _fill_vals[-1]
    
    
    if verbose:
        fig = plt.figure(figsize = figsize)
        lmap(lambda x: plt.axvline(x, color='r'), _centers.k.dropna())
        lmap(lambda x: plt.axvline(x, color='g'), _centers.means)
        _ = sns.distplot(_show_intensities, kde=False)
        
        fig2 = plt.figure(figsize = figsize)
        fig2.add_subplot(1, 2, 1)
        plt.imshow(img, cmap = 'gray')
        plt.title('Original')
        fig2.add_subplot(1, 2, 2)
        plt.imshow(dst, cmap = 'gray')
        plt.title(f"Threshold ({groups} groups)")
        
        
        
    return dst
##

def ref_region(
    img: np.ndarray,
    selem: Any = disk(5),
    sigma: int = 3,
    opening_se: np.ndarray = np.ones((10, 10)),
    closing_se: np.ndarray = np.ones((5, 5)),
    verbose: bool = False
):
    """
    """
    
    # Perform histogram equalisation :
    _img_eq = rank.equalize(img, selem=selem)
    
    # Perform edge detection :
    _edges = canny(_img_eq, sigma=3)
    _filled = ndi.binary_fill_holes(_edges)
    
    # Morphological processing :
    _eroded = utils.closing(
        utils.opening(np.float64(_filled), opening_se), closing_se
    )
    
    if verbose:
        utils.side_by_side(img1, img_eq, title1="Original", title2="Histogram Equalised")
        #plt.title('Lol')
        utils.side_by_side(img_eq, filled, title1="Histogram Equalised", title2="Canny Edge Detection + Filled image")
        #plt.title('Lal')
        utils.side_by_side(filled, eroded, title1="Canny Edge Detection + Filled image", title2="Opening, closing")
        #plt.title('Lel')
        
    return eroded


# In[6]:


ls images/


# In[7]:


cwd  = os.path.abspath('.')
path = os.path.join(cwd, 'images')
pattern = os.path.join(path, '*flujo.png')
files = glob.glob(pattern)
files


# Todas nuestras imágenes de interés contienen la cadena de caracteres 'flujo.png'.

# In[8]:


llaves = lmap(lambda x: os.path.split(x)[-1], files)


# In[9]:


mangueras = {
    f"{nombre}": imread(file) for file, nombre in zip(files, llaves) 
}


# In[10]:


intensities = pd.core.frame.DataFrame({
    key: mangueras[key].flatten() for key in mangueras.keys()
})


# In[9]:


# SUPER SLOW ! 
# Do not run !
sns.pairplot(intensities)


# Podemos observar una gran correlación entre las intensidades de todas las imágenes.

# In[11]:


for i in intensities:
    sns.distplot(intensities[i],  kde=False)


# Nótese lo similares que son las distribuciones de las intensidades, independientemente de la intensidad del flujo.

# In[16]:


mangueras_segmentadas = {
    key: auto_segment(mangueras[key]) for key in mangueras.keys()
}


# Aquí segmentamos automáticamente la región de la manguera, gracias al gran contraste que existe entre éste nuestro ente de interés y el fondo (muy claro el primero, oscuro el segundo).
# 
# Usamos la función que diseñamos : ```auto_segment()```

# In[18]:


for nombre in mangueras.keys():
    utils.side_by_side(
        mangueras[nombre], mangueras_segmentadas[nombre], 
        title1=nombre, title2=f"{nombre} : manguera segmentada"
    )


# Aquí podemos observar las imágenes con su respectiva máscara de segmentación.

# In[35]:


region_ref1 = {
    key: auto_segment(mangueras[key], groups=3) for key in mangueras.keys()
}


# In[36]:


for nombre in mangueras.keys():
    utils.side_by_side(
        mangueras[nombre], region_ref1[nombre], 
        title1=nombre, title2=f"{nombre} : región de referencia segmentada"
    )


# Aquí podemos observar que la referencia es más difícil de segmentar en función de las intensidades. 
# 
# La función fue llamada indicando que se buscaba una imagen trinaria ```auto_seg(img, groups=3)```
# Se esperaba que esto permitiese segmentar la **región referencia** ya que ésta muestra una intensidad mayor a la del fondo pero menor a la de la manguera.
# 
# Tal vez quitando la región de la manguera (la de mayor intensidad) sea más fácil segmentar automáticamente la **región referencia**.

# In[37]:


sin_manguera = {
    key: mangueras[key] * np.uint8(1.0 - mangueras_segmentadas[key])
    for key in mangueras_segmentadas.keys()
}
plt.imshow(sin_manguera[llaves[0]], cmap='gray')


# Nótese que la imagen muestra en negro la región que antes mostraba la mayor intensidad.

# In[38]:


sin_manguera = {
    key: mangueras[key] * np.uint8(1.0 - mangueras_segmentadas[key])
    for key in mangueras_segmentadas.keys()
}


# In[65]:


region_ref2 = {
    key: auto_segment(sin_manguera[key], groups=2, nonzero=True) for key in sin_manguera.keys()
}


# In[66]:


for nombre in sin_manguera.keys():
    utils.side_by_side(
        sin_manguera[nombre], region_ref2[nombre], 
        title1=nombre, title2=f"{nombre} : región de referencia segmentada"
    )


# Aún teniendo la región de la manguera oscurecida, la función ```auto_seg()``` no permite segmentar la **región referencia** de forma automática. Esto podría atribuirse a que la forma del histograma de las ***imágenes con la manguera oscurecida*** sigue mostrando dos cúmulos principales como se muestra a continuación.
# 
# Sin embargo, debe notarse que la funcción ```auto_seg(.., nonzero=True)``` fue llamada con el parámetro ```nonzero=True```, lo que hace que la funcón ignore las entradas que valen 0 al momento de calcular los centros de los grupos.
# 
# Si se desea una visualización más detallada del funcionamiento de este parámetro, se recomienda correr este código, en dos celdas por separado para observar el efecto del parámetro ```nonzero``` :
# ```python
# region_ref2 = {
#     key: auto_segment(sin_manguera[key], groups=2, nonzero=True, verbose=True) for key in sin_manguera.keys()
# }
# ```
# por
# ```python
# region_ref2 = {
#     key: auto_segment(sin_manguera[key], groups=2, nonzero=False, verbose=True) for key in sin_manguera.keys()
# }
# ```

# In[44]:


sns.distplot(sin_manguera[llaves[2]].flatten())


# In[69]:


region_ref3 = {
    key: auto_segment(sin_manguera[key], groups=3, nonzero=True) for key in sin_manguera.keys()
}


# In[70]:


for nombre in sin_manguera.keys():
    utils.side_by_side(
        sin_manguera[nombre], region_ref3[nombre], 
        title1=nombre, title2=f"{nombre} : región de referencia segmentada"
    )


# In[89]:


edges = canny(mangueras[llaves[0]] /255.)
fill_coins = ndi.binary_fill_holes(edges)


# In[158]:


verbose = False

if verbose:
    for img1 in mangueras.values():
        ref_region(img1, verbose=True)


# In[164]:


region_ref4 = {
    key: ref_region(mangueras[key]) for key in mangueras.keys()
}


# In[167]:


for nombre, imagen in zip(region_ref4.keys(), region_ref4.values()):
    plt.figure()
    plt.imshow(np.uint8(imagen), cmap="gray")
    plt.title(nombre)


# In[110]:


label_image, n_objs = label(filled, return_num=True)


# In[112]:


fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(label_image)
print(n_objs)


# In[ ]:





# In[34]:


mangueras_segmentadas[llaves[0]]


# In[24]:


for i in reg_ref_segmentadas.keys():
    mask = np.nonzero(reg_ref_segmentadas[i] > K)
    reg_ref_segmentadas[i][mask] = 0


# In[76]:


for nombre in reg_ref_segmentadas.keys():
    plt.figure()
    plt.imshow(reg_ref_segmentadas[nombre], cmap="gray")
    plt.title(nombre)


# In[79]:


sns.distplot(reg_ref_segmentadas['altoflujo.png'].flatten())


# In[80]:


intensities2 = pd.core.frame.DataFrame({
    key: reg_ref_segmentadas[key].flatten() for key in reg_ref_segmentadas.keys()
})


# In[101]:


"""kmeans2 = KMeans(
    n_clusters=2, 
    random_state=0, 
    verbose=False
).fit(
    intensities2.
)"""


# In[93]:


seg = mangueras[llaves[0]] * 


# In[98]:


seg.max(), mangueras[llaves[0]].max(), mangueras_segmentadas[llaves[0]].max()


# In[99]:


utils.side_by_side(mangueras[llaves[0]], seg)


# In[100]:


utils.side_by_side(mangueras_segmentadas[llaves[0]], seg)


# In[108]:


mangueras_segmentadas[llaves[0]].dtype


# In[133]:


_tmp_img = mangueras[llaves[0]]
utils.side_by_side(_tmp_img, auto_segment(_tmp_img, groups=2))


# In[117]:


_tmp_img.flatten().reshape(-1, 1)


# In[123]:


y = [[1, 2], [3, 4]]


# In[125]:


print(*y)


# In[284]:


_tmp_img = mangueras[llaves[0]]
mask = auto_segment(_tmp_img, groups=2)
sns.distplot(mask.flatten())
utils.side_by_side(_tmp_img, mask)


# In[196]:





# In[198]:


sns.distplot(mask.flatten(), kde=False)


# In[170]:


mask[ mask == 63 ].shape


# In[177]:


2*512*512


# In[181]:


x= np.linspace(0, 1)


# In[182]:


x[-]


# In[184]:


x[::-1] 


# In[216]:


pd.core.frame.DataFrame({"h": [np.nan, 1, np.nan, 3]}).fillna(0)


# In[ ]:




