# coding: utf-8

# # División de Ciencias e Ingenierías de la Universidad de Guanajuato
# ## Fundamentos de procesamiento digital de imágenes
# ## TAREA : Funciones de filtrado en frecuencia
# ### Profesor : Dr. Arturo González Vega
# ### Alumno : Gustavo Magaña López

import copy
from typing import Tuple, List, NoReturn

import numpy as np
import scipy.fftpack as F
import scipy.io as io

import cv2
#import matplotlib.image as img

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm

import matplotlib
import skimage
import skimage.morphology
import skimage.filters


eps = np.finfo(float).eps
eps.setflags(write=False)

def img_surf(
       image: np.ndarray,
    colormap: matplotlib.colors.LinearSegmentedColormap = cm.viridis,
   la_figura: matplotlib.figure.Figure = None
) -> None:
    """
    """

    if type(la_figura) is not None:
        fig  = plt.figure()
        ax   = fig.gca(projection='3d')
    else:
        fig = la_figura
        ax  = fig.gca(projection='3d')

    x, y = list(map(lambda x: np.arange(0, x), image.shape))
    X, Y = np.meshgrid(x, y)
    #U, V = fourier_meshgrid(image)
    #print(f'Shapes X:{X.shape}\n Y:{Y.shape}\n Z:{Z.shape}')

    surf = ax.plot_surface(X, Y, image.T, cmap=colormap,
                            linewidth=0, antialiased=False)
    return surf
    # plt.show()
##

def img_fft(image: np.ndarray, shift: bool = True) -> np.ndarray:
    """
        Ejecutar una Transformada de Fourier visualizable con matplotlib.pyplot.imshow() .
        
        Basado en un snippet encontrado en :
        https://medium.com/@y1017c121y/python-computer-vision-tutorials-image-fourier-transform-part-2-ec9803e63993
        
        Parámetros :
                image : Imagen, representada como un arreglo de numpy (numpy.ndarray)
                shift : Booleano que indica si debe ejecutarse la traslación de la imagen e
                        en el espacio de frecuencia.
    """
    _X = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
    if shift:
        _X_shift = np.fft.fftshift(_X)
    _X_complex = _X_shift[:,:,0] + 1j*_X_shift[:,:,1]
    _X_abs = np.abs(_X_complex) + 1 # Evitar que el logaritmo reciba 0 como argumento.
    _X_bounded = 20 * np.log(_X_abs)
    _X_img = 255 * _X_bounded / np.max(_X_bounded)
    _X_img = _X_img.astype(np.uint8)
    
    return _X_img
##

def fft_viz(
    image: np.ndarray, 
    shift: bool = True, 
   newfig: bool = False
) -> None:
    """
        Ver la transformada de fourier de una imagen.
    """
    if newfig:
        plt.figure()
    return plt.imshow(img_fft(image, shift=shift), cmap='gray')
##

def pre_fft_processing(
     image: np.ndarray,
       top: int = None,
    bottom: int = None,
      left: int = None,
     right: int = None,
borderType: int = cv2.BORDER_CONSTANT,
) -> np.ndarray:
    """
        Esta función lleva a cabo el proceso de 'padding', necesario antes 
        de aplicar un filtrado en el espacio frecuencial (de Fourier).
        
        Básicamente un wrapper para :
            cv2.copyMakeBorder( image, top, bottom, left, right, borderType)
            
        Donde el tipo de frontera (borderType) puede ser alguno de los siguientes :
            cv2.BORDER_CONSTANT
            cv2.BORDER_REFLECT
            cv2.BORDER_REFLECT_101
            cv2.BORDER_DEFAULT
            cv2.BORDER_REPLICATE
            cv2.BORDER_WRAP
        
        Si faltan uno o más de los parámetros necesarios para llamar cv2.copyMakeBorder(), 
        se implementará un 'padding' por defecto en función de los valores calculados por
        cv2.getOptimalDFTSize().
        
    ##  Parámetros :
                image : Una imagen en blanco y negro, es decir un arreglo bidimensional de numpy.
                  top : Número entero representando el número de pixeles que se agegarán en el margen superior.
               bottom : idem. para el margen inferior. 
                 left : idem. para el margen izquierdo.
                right : idem. para el margen derecho.
           borderType : Alguno de los mencionados anteriormente en el Docstring.


                
    ##  Regresa :
                nimg  : imagen con 'padding'
    """
    override = all(map(lambda x: x if x != 0 else True, [top, bottom, left, right, borderType]))
    if override:
        nimg = cv2.copyMakeBorder(image, top, bottom, left, right, bordertype)
    else:
        rows, cols = image.shape
        nrows, ncols = list(map(cv2.getOptimalDFTSize, image.shape))
        right = ncols - cols
        bottom = nrows - rows
        bordertype = cv2.BORDER_CONSTANT #just to avoid line breakup in PDF file
        nimg = cv2.copyMakeBorder(image, 0, bottom, 0, right, bordertype, value = 0)
    
    return nimg
##

def fft2(
    image: np.ndarray,       
      top: int = None,
   bottom: int = None, 
     left: int = None,
    right: int = None,
    borderType: int = None,
) -> np.ndarray:
    """
    
    Execute:
        x = pre_fft_processing(image, top=top, bottom=bottom, left=left, right=right, borderType=borderType)
        return cv2.dft(np.float32(x),flags=cv2.DFT_COMPLEX_OUTPUT)
        
    Call the cv2's dft, which is supposed to be considerably faster than numpy's implementation.

    See help(pre_fft_processing) for futher details on the preprocessing stage.
    
    """
    nimg = pre_fft_processing(image, top=top, bottom=bottom, left=left, right=right, borderType=borderType)
    dft2 = cv2.dft(np.float32(nimg),flags=cv2.DFT_COMPLEX_OUTPUT)
    
    return dft2
##

def ImPotencia(image: np.ndarray) -> float:
    """
        Calcula la potencia de acuerdo al teorema de Parseval.
    """
    _F = np.fft.fft2(image)
    return np.sum(np.abs(_F)**2) / np.prod(_F.shape)
##

def fourier_meshgrid(image: np.ndarray) -> Tuple[np.ndarray]:
    """
        Genera los arreglos bidimensionales U y V necesarios para poder hacer tanto
        filtrado en frecuencias como la visualización de imágenes en forma de superficies.
        Esto se hace mapeando las intensidades a los valores que tomará la función en el eje
        Z, dados los valores de X y Y que son las coordenadas de los pixeles.
        
    
    Parámetros :
        imagen : Arreglo bidimensional de numpy (numpy.ndarray), es decir una imagen.
        
    Regresa :
        (U, V) : Tuple contieniendo dos arreglos bidimensionales de numpy (numpy.ndarray)
    """
    M, N = image.shape
    u, v = list(map(lambda x: np.arange(0, x), image.shape))
    idx, idy = list(map(lambda x, y: np.nonzero(x > y/2), [u, v], image.shape))
    u[idx] -= M
    v[idy] -= N
    V, U = np.meshgrid(v, u)
    
    return U, V
##

def fourier_distance(U: np.ndarray, V: np.ndarray, centered: bool = True, squared: bool = True) -> np.ndarray:
    """
        Calcula la distancia euclidiana de los puntos de una malla (meshgrid), respecto al centro.
        Por defecto desplaza el centro (distancia 0) al centro de la matriz.
        Asimismo devuelve la distancia al cuadrado puesto que en ocaciones dicho cálculo se hace después
        y calcular la raíz y después elevar al cuadrado sería sólo perder tiempo de cómputo.
        
    Parámetros :
    
                U : Arreglo bidimensional de numpy (numpy.ndarray). 
                V : Idem.
         centered : Booleano indicando si se desea la distancia centrada, 
                    es decir ejecutar np.fft.fftshift(Distancia) una vez calculada
                    la matriz de distancias. 
                        True por defecto.
                        
          squared : Booleano indicando si se desea la distancia al cuadrado 
                    o la distancia euclidiana clásica.
                        True por defecto.
    
    Regresa :
               _d : Matriz con las distancias euclidianas, 
                    de cada coordenada respecto al centro.
    """
    _d = U**2 + V**2
    if not squared:
        _d = np.sqrt(_d)
    if centered:
        _d = np.fft.fftshift(_d)
    
    return _d
##

def _param_check(kind: str, Do: int) -> bool:
    """
        Para reducir la redundancia en el cuerpo de las funciones, 
        esta función verifica que :
        1.- La formulación especificada 'kind' sea válida.
            i.e. Alguna de las siguientes :
                'low', 'lowpass', 'low pass',
                'high', 'highpass', 'high pass',
                'bandpass', 'bandstop', 
                'band pass', 'band stop',
                'bandreject', 'band reject',
                'notchpass', 'notchreject',
                'notch pass', 'notch reject'
        2.- Que el parámetro `frecuencia de corte`, es decir
            Do o sigma, sea positivo.
    
    Parámetros :
            kind : string, tipo de filtro.
              Do : int, valor de la frecuencia de corte (distancia en el espacio de Fourier)
              
    Regresa :
            True  : Si se cumplen ambas condiciones.
            False : Si no. 
        
    """
    _kinds = [
        'low', 'high', 'lowpass', 'highpass', 
        'low pass', 'high pass',
        'bandpass', 'bandstop', 
        'band pass', 'band stop',
        'bandreject', 'band reject',
        'notchpass', 'notchreject',
        'notch pass', 'notch reject'
    ]
    kind = kind.lower()
    _kind_check = kind in _kinds
    _dist_check = Do > 0
    
    return _kind_check and _dist_check
##

def _param_check2(form: str, Do: int) -> bool:
    """
        Para reducir la redundancia en el cuerpo de las funciones, 
        esta función verifica que :
        1.- La formulación de filtro especificada, 'form' se válida
            i.e. Alguna de las siguientes :
            
            'ideal', 'btw', 'butterworth', 'gauss', 'gaussian'

        2.- Que el parámetro `frecuencia de corte`, es decir
        
            'Do' o 'sigma', sea positivo.
    
    Parámetros :
            kind : string, tipo de filtro.
              Do : int, valor de la frecuencia de corte (distancia en el espacio de Fourier)
              
    Regresa :
            True  : Si se cumplen ambas condiciones.
            False : Si no. 
        
    """
    _forms = ['ideal', 'btw', 'butterworth', 'gauss', 'gaussian']

    assert type(form) is str, f'form is of type {type(form)}, should be str.'
    assert type(Do) is int, f'Do is of type {type(Do)}, should be int.'
    
    form = form.lower()
    _form_check = form in _forms
    _dist_check = Do > 0
    
    return _form_check and _dist_check
##



def kernel_lowpass(
    image: np.ndarray, 
       Do: int = 15,
     form: str = 'ideal',
        n: int = None
) -> np.ndarray:
    """
        Diseña un filtro pasa bajos.
    """
    
    form = form.lower()
    assert _param_check2(form, Do), 'Formulación del filtro o frecuencia de corte inválidas.'
    
    U, V = fourier_meshgrid(image)
    D = fourier_distance(U, V)
    H = np.zeros_like(D)
    
    if form == 'ideal':
        _mask = np.nonzero(D <= Do)
        H[_mask] = 1.0
    elif 'gauss' in form:
        H = np.exp( (-1.0 * D) / (2.0 * Do**2) )
    else:
        if n is None:
            n = 1
        else:
            assert type(n) is int, f'n debe ser de la clase int, no {type(n)}'
            assert n in range(1, 10+1), f'n = {n} no es válido. n debe estar en [1, 10]'
        H = 1.0 / ( 1.0 + (D / Do**2)**n )
    
    return H
##

def kernel_highpass(
    image: np.ndarray,
       Do: int = 15,
     form: str = 'ideal',
        n: int = None
) -> np.ndarray:
    """
        Diseña un filtro pasa altos.
    """
    
    return 1.0 - kernel_lowpass(image, Do=Do, form=form, n=n)
##

def kernel_band_reject(
    image: np.ndarray,
       Do: int = 50,
        w: int = 15,
      wc1: int = None,
      wc2: int = None,
     form: str = 'ideal',
        n: int = 1
) -> np.ndarray:
    """
        Diseña un filtro de rechazo de banda.
    """
    
    if wc1 and wc2:
        assert type(wc1) is int and type(wc2) is int,\
            f'Argumentos wc1 y wc2 deben ser de tipo entero, no : {type(wc1)}, {type(wc2)}'
        assert wc1 < wc2,\
            f'Valores wc1 = {wc1}, wc2 = {wc2} no cumplen wc1 < wc2.'
        Do = np.ceil( (wc1 + wc2) / 2 )
        w  = wc2 - wc1
    else:
        assert type(Do) is int and type(w) is int,\
            f'Argumentos Do y w deben ser de tipo entero, no : {type(wc1)}, {type(wc2)}'

    assert _param_check2(form, int(Do)), 'Formulación de filtro inválida.'
    U, V = fourier_meshgrid(image)
    D = fourier_distance(U, V)
    H = np.ones_like(D)

    form = form.lower()
    if form == 'ideal':
        _mask = np.nonzero( (D >= Do - w/2) & (D <= Do + w/2))
        H[_mask] = 0.0
    elif form == 'btw' or form == 'butterworth':
        assert type(n) is int, f"n debe ser de tipo 'int', no {type(n)}"
        assert n in range(1, 10+1), f'n (={n}) debe estar en [1, 10]'
        H = 1.0 / (1.0 + ( w**2 * D / (D - Do**2 + eps)**2 )**n )
    elif 'gauss' in form:
        H = 1.0 - np.exp(-1.0 * (D - Do**2)**2 / (w**2 * D) )
    else:
        pass

    return H
##

def kernel_band_pass(
    image: np.ndarray,
       Do: int = 50,
        w: int = 15,
      wc1: int = None,
      wc2: int = None,
     form: str = 'ideal',
        n: int = 1
) -> np.ndarray:
    """
        Diseña un filtro pasa bandas.
    """
    return 1.0 - kernel_band_reject(image, Do=Do, w=w, wc1=wc1, wc2=wc2, form=form, n=n)
##

def distance_meshgrid_2D(image: np.ndarray) -> np.ndarray:
    """
        Genera una proyección visualizable de las distancias
        dentro de una meshgrid, respecto al centro.
    """
    _U, _V = fourier_meshgrid(image)
    _D  = fourier_distance(_U, _V)
    _H  = np.zeros_like(_D)
    _dd = (_D / _D.max())* 255
    _di = np.uint8(_dd)
    return _dd
##

def master_kernel(
    image: np.ndarray,
       Do: int = 50,
        w: int = 15,
      wc1: int = None,
      wc2: int = None,
     kind: str = 'low',
     form: str = 'ideal',
   center: Tuple[int] = (0, 0),
        n: int = 1
) -> np.ndarray:
    """
        Dados:
            una imagen
            un tipo de filtro (lowpass, highpass, bandpass, bandreject, notch)
            una folrmulación
            los parámetros de diseño necesarios
        
        Calcula (diseña) un kernel de acuerdo a todas las especificaciones dadas.
    """

    assert _param_check(kind, Do), f'Invalid filter kind : `{kind}`, see help(mfilt_funcs._param_check)'
    assert _param_check2(form, Do), f'Invalid filter formulation `{form}` see help(mfilt_funcs._param_check2)'

    H = np.zeros_like(image)

    if 'low' in kind:
        H = kernel_lowpass(image, Do=Do, form=form, n=n)
    elif 'high' in kind:
        H = kernel_highpass(image, Do=Do, form=form, n=n)
    elif 'band' in kind:
        if ('reject' in kind) or ('stop' in kind):
            H = kernel_band_reject(image, Do=Do, w=w, wc1=wc1, wc2=wc2, form=form, n=n)
        else:
            H = kernel_band_pass(image, Do=Do, w=w, wc1=wc1, wc2=wc2, form=form, n=n)
    elif 'notch' in kind:
        _forma = 0
        _pasa  = 0
        if 'ideal' in form:
            _forma = 0 
        elif 'gauss' in form:
            _forma = 1
        else:
            _forma = 2
        if 'pass' in kind:
            _pasa = 1
        H = kernel_notch(image, Do, centro=center, tipo=_forma, pasa=_pasa, n=n)
    
    return H
##
    
def filtra_maestra(
    image: np.ndarray,
       Do: int = 50,
        w: int = 15,
      wc1: int = None,
      wc2: int = None,
     kind: str = 'low',
     form: str = 'ideal',
   center: Tuple[int] = (0, 0),
        n: int = 1
) -> np.ndarray:
    """
        Diseña y aplica un filtro.
        Envoltura para master_kernel(*args, **kw)
        
        return FFT

    """

    kernel = master_kernel(
           image,
           Do = Do,
            w = w,
          wc1 = wc1,
          wc2 = wc2,
         kind = kind,
         form = form,
       center = center,
            n = n
    ) 

    transformada = np.fft.fftshift(np.fft.fft2(image))
    aplico_filtro = kernel * transformada
    img_filtrada = np.real(np.fft.ifft2(np.fft.ifftshift(aplico_filtro)))

    return img_filtrada
##

def __FiltraGaussiana(image: np.ndarray, sigma: float, kind: str = 'low') -> np.ndarray:
    """
        DO NOT USE THIS FUNCTION !
    """
    kind   = kind.lower()
    _kinds = ['low', 'high', 'lowpass', 'highpass']
    if kind not in _kinds:
        raise Exception(f'Error : Tipo desconocido de filtro \"{kind}\".\n Tipos disponibles : {_kinds}')
    
    H  = kernel_gaussiano(image=image, sigma=sigma, kind=kind)
    _F = np.fft.ifftshift(
            np.fft.fft2(image)
    )
    G  = H * _F
    g  = np.real( np.fft.ifft2( np.fft.ifftshift(G) ))
    
    # Recortamos la imagen a su tamaño original, de ser requerido.
    g = g[:image.shape[0], :image.shape[1]]  
        
    return g  
##

def filtro_disco(image: np.ndarray, radius: int = 5) -> np.ndarray:
    """
    
    """
    _circle = skimage.morphology.disk(radius)
    _filtered = skimage.filters.rank.mean(copy.deepcopy(image), selem=_circle)
    return _filtered
##


def imagen_dft(imagen):
    """Esta función solo sirve con imágenes BGR,
    aunque sean a escala de grises. Usa cv2.imread(imagen, 0)"""
    dft = cv2.dft(np.float32(imagen), flags = cv2.DFT_COMPLEX_OUTPUT) # Transformada de la imagen
    dft_shift = np.fft.fftshift(dft) # Centramos la transformada
    magnitud = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) # Magnitud del espectro
    # Regresar a imagen int
    cota = 20 * np.log(magnitud)
    img_transf = 255 * cota / np.max(cota)
    img_transf = img_transf.astype(np.uint8)
    
    return img_transf
##

def kernel_ideal(M, N, centro, d0):
    u_k = centro[0]
    v_k = centro[1]
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = np.where(D_k <= d0**2, 0, 1) # Primer pasaaltos
    H_mk = np.where(D_mk <= d0**2, 0, 1) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel
##

def kernel_gaussiano(M, N, centro, d0):
    u_k = centro[0]
    v_k = centro[1]
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = 1 - np.exp(-(0.5 / d0**2) * D_k) # Primer pasaaltos
    H_mk = 1 - np.exp(-(0.5 / d0**2) * D_mk) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel
##

def kernel_butterworth(M, N, centro, d0, n):
    u_k = centro[0]
    v_k = centro[1]
    u = np.arange(M)
    v = np.arange(N)
    V, U = np.meshgrid(v, u)
    
    D_k = np.square(U - 0.5 * M - u_k) + np.square(V - 0.5 * N - v_k)
    D_mk = np.square(U - 0.5 * M + u_k) + np.square(V - 0.5 * N + v_k)
    H_k = np.divide(D_k**n, D_k**n + d0**(2*n)) # Primer pasaaltos
    H_mk = np.divide(D_mk**n, D_mk**n + d0**(2*n)) # Segundo pasaaltos
    kernel = H_k * H_mk
    
    return kernel
##

def kernel_notch(img, d0, centro = (0, 0), tipo = 0, pasa = 0, n = 1):
    """
    Filtro notch. 
    tipo = 0 para ideal, 1 para gaussiano y cualquier otro valor para butterworth.
    pasa = 0 para notchreject, 1 para notchpass.
    centro y radio son los del notch. notch simétrico automático.
    Especificar n solo para butterworth
    """
    
    M, N = img.shape
    
    if tipo == 0:
        kernel_prov = kernel_ideal(M, N, centro, d0)
    elif tipo == 1:
        kernel_prov = kernel_gaussiano(M, N, centro, d0)
    else:
        kernel_prov = kernel_butterworth(M, N, centro, d0, n)
    
    kernel = pasa + (-1.0)**pasa * kernel_prov
    return np.float64(kernel_prov)
##

