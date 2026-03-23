'''
##################################################################################################
#--------------------------------------------- Deep TIQ -----------------------------------------# 
#------------------------------------------------------------------------------------------------# 
#---------- A python library for deep learning based image segmentation and analysis ------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------ Adrien Hallou ---------------------------------------#
#------------------------------------------------------------------------------------------------#
#-------------------------------------- University of Oxford ------------------------------------#
#------------------------------------------------------------------------------------------------#                         
#-------------------------------------------- 2018-2025------------------------------------------#
#------------------------------------------------------------------------------------------------#
##################################################################################################
 
 
##################################################################################################           
#------------------------------------- Deep TIQ Segmentation ------------------------------------#
#------------------------------------------------------------------------------------------------#
#------------------------------------------ Adrien Hallou ---------------------------------------#
#------------------------------------------------------------------------------------------------#
#----------- Utilities functions for image segmentation (training, prediction,evaluation) -------#
#------------------------------------------------------------------------------------------------#
##################################################################################################
'''

############################################ Python libraries ############################################

import os, glob

import numpy as np

import pandas as pd

import scipy as sp
from scipy import ndimage
from scipy.spatial.distance import jaccard
from scipy.optimize import linear_sum_assignment

import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import colorcet as cc

import skimage
from skimage import io, color, img_as_float
from skimage.util import img_as_ubyte
from skimage.transform import resize 
from skimage.measure import find_contours, label, regionprops_table
from skimage.morphology import square, binary_opening, remove_small_objects
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.segmentation import clear_border, relabel_sequential

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, jaccard_score, matthews_corrcoef, roc_auc_score,precision_recall_curve

import cv2
from cv2 import COLOR_GRAY2RGB, cvtColor, addWeighted

import SimpleITK as sitk

import PIL
from PIL import Image, ImageOps

import patchify
from patchify import patchify, unpatchify

import latextable
from tabulate import tabulate
from texttable import Texttable

import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras import backend as K

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.metrics import binary_accuracy,MeanIoU

from tensorflow.python.ops import array_ops, math_ops

from tensorflow.keras.utils import Sequence

from tensorflow.keras.models import Model, save_model, load_model

from tensorflow.keras.losses import binary_crossentropy

from tensorflow.keras.layers import Input, UpSampling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import AveragePooling2D,MaxPooling2D
from tensorflow.keras.layers import concatenate,Concatenate

import segmentation_models as sm
from segmentation_models import get_preprocessing

import patchify
from patchify import patchify, unpatchify

################################################ Training ################################################

###### Image pre-processing functions ###### 

def load_images(path, folder, file_ext, re_size=False, new_shape=None, verbose_mode=True):
    ''' Load image, binary mask or weight map patches. 
    ----------
    Arguments
    ----------
            path : String - Path to where the files are stored on drive
            folder : String - Folder in which the files are stored
            file_ext : String - File extension (e.g. '.tif')
            re_size : Boolean - Resize or not loaded patches 
            new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches.
    ----------
    Returns
    ----------
            images :  Array - List of loaded images.
    '''
    filenames = [i for i in os.listdir( path + folder ) if i.endswith(file_ext)] # Load filenames
    filenames.sort() # Sort filenames
    if verbose_mode == True:
        print('Images: ' + str(filenames))
        print( 'Number of images: ' + str( len(filenames)) )
    else:
        pass
    if re_size is False:
        images = [ img_as_ubyte( io.imread( path + folder + i ) ) for i in filenames ] # Create array storing images
    else:
        images = [resize( io.imread( path + folder + i ), output_shape=new_shape, anti_aliasing=True ) for i in filenames ]   
    return (images)

def weight_map(mask, l_0, sigma_bal, sigma_sep):
    ''' Create pixel-wise weight map correcting for class imbalance and undersegmentation
        of touching foreground instances setting higher weights for pixel separating them.
    ----------
    Arguments
    ----------
              mask : Array - binary mask.
              l_0 : Float - strength of instance separation - Default = 50
              sigma_bal : Float - S.D of the gaussian kernel used to compute smooth cut-off between foreground and background - Default = 10
              sigma_sep : Float - S.D of the gaussian kernel used to compute spatial weights - Default = 5.
    ----------
    Returns
    ----------
            weight_map : Array - 2D array of same dimensions as input binary mask and containing pixel-wise weights (floats).
    ''' 
    # Normalise mask values on [0,1] instead of [0,255]
    mask[mask > 0.5] = 1
    mask[mask < 0.5] = 0
    # Case 1: The mask has only one contiguous class (i.e only background or only foreground)
    if len(np.unique(mask)) == 1: 
      weight_map = (np.ones(mask.shape, dtype=float) * 0.5)
    else:
      # Compute distance maps.
      segs, num_segs = ndimage.label(mask) # segs = mask with foreground instances labelled with different integer & num_segs = number of detected foreground instances 
      inst_lbl = range(1, np.max(segs)+1) # List of instance labels starting from 1 (0 is background)
      distmaps = np.stack([ndimage.distance_transform_edt(segs != l) for l in inst_lbl]) # Compute distances of all pixels to all foreground instances
      distmaps = np.sort(distmaps, axis=0)[:2]
      d_1 = distmaps[0] # Distances to first nearest labelled foreground intances. 
      d_2 = distmaps[1] # Distances to second nearest labelled foreground intances.
      # Compute w_b the class-balanced weight map 
      w_b = np.zeros(mask.shape, dtype=float) 
      F_fp = (float(np.count_nonzero(mask)) / w_b.size)
      w_0 = F_fp # Background pixels (= 0) class-balanced pixel weights       
      w_1 = (1 - F_fp)   # Foreground pixels (= 1) class-balanced pixel weights 
      w_b[mask == 0] = w_0 
      w_b[mask == 1] = w_1
      # Compute w_c a smooth cut-off mask for the boundary of foreground and background objects
      w_c = (1 - w_0) * np.exp((-1 * (d_1)** 2) / (2 * (sigma_bal ** 2)))
      w_c[mask == 1] = 0.0 # Ensure that only background pixels are weighted.
      # Case 2: The mask has 1 foreground instance.
      if np.max(segs) == 1:
        weight_map =  w_b + w_c
      # Case 3: The mask has multiple foreground instances. Weights of background pixels between instances are computed.
      else:
        # Compute the distance-weighted map w_e
        w_e = l_0 * np.exp((-1 * (d_1 + d_2) ** 2) / (2 * (sigma_sep ** 2))) 
        w_e[mask == 1] = 0.0 # Ensure that only background pixels are weighted. 
        weight_map = ( w_b + w_c + w_e )
    return (weight_map)

def compute_weight_maps(in_path, in_folder, file_ext, l_0, sigma_bal, sigma_sep, out_path, out_folder, re_size=False, new_shape=None):
    ''' Compute and save pixel-wise weight maps.
    ----------
    Arguments
    ----------
            in_path : String - Path to where the mask are stored on drive
            in_folder : String - Folder in which the masks are stored on drive
            file_ext : String - File extension (e.g. '.tif')
            l_0 : Float - Strength of instance separation - Default = 50
            sigma_bal : Float - S.D of the gaussian kernel used to compute smooth cut-off between foreground and background - Default = 10
            sigma_sep : Float - S.D of the gaussian kernel used to compute spatial weights - Default = 5
            out_path : String - Path to where the files are stored on drive
            out_folder : String - Folder in which the files are stored
            re_size : Boolean - Resize or not loaded patches 
            new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches.
    ----------
    Returns
    ----------
            weight_maps : Array - List of 2D array of same dimensions as input binary mask and containing pixel-wise weights (floats).
    '''
    # Import masks
    filenames = [i for i in os.listdir( in_path + in_folder ) if i.endswith(file_ext)] # Load filenames
    filenames.sort() # Sort filenames
    if re_size is False:
        masks = [ img_as_ubyte( io.imread( in_path + in_folder + i ) ) for i in filenames ] # Create array storing input or output patches
    else:
        masks = [resize( io.imread( in_path + in_folder + i ), output_shape=new_shape, anti_aliasing=True ) for i in filenames ]  
    # Compute weight maps
    weight_maps = [ weight_map(x, l_0, sigma_bal, sigma_sep) for x in masks ]
    print( 'Number of weight maps computed: ' + str( len(weight_maps)) )
    # Save weight maps
    for i in range(0,len(weight_maps)):
      io.imsave(out_path + out_folder + 'Weight_Map_l0_'+ str(l_0) + '_sigma-bal_' + str(sigma_bal) + '_sigma-sep_' + str(sigma_sep) + '_' + str(filenames[i]), weight_maps[i], plugin='tifffile')
    print('All pixel-wise weight maps have been saved.') 
    return (weight_maps)

def load_weight_maps(path, folder, file_ext, re_size=False, new_shape=None):
    ''' Load pixel-wise weight maps.
    ----------
    Arguments
    ----------
            path : String - Path to where the files are stored on drive
            folder : String - Folder in which the files are stored
            file_ext : String - File extension (e.g. '.tif')
            re_size : Boolean - Resize or not loaded patches 
            new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches.
    ----------
    Returns
    ----------
            images :  Array - List of loaded images.
    '''
    filenames = [i for i in os.listdir( path + folder ) if i.endswith(file_ext)] # Load filenames
    filenames.sort() # Sort filenames
    print('Weight maps: ' + str(filenames))
    print( 'Number of weight maps: ' + str( len(filenames)) )
    if re_size is False:
        maps = [ io.imread( path + folder + i ) for i in filenames ] # Create array storing maps
    else:
        maps = [resize( io.imread( path + folder + i ), output_shape=new_shape, anti_aliasing=True ) for i in filenames ]   
    return (maps)
    
def random_patches( images, masks, weights, num_patches, shape, bck_th):
    ''' Create a given number of random patches of given size (x,y) out of input 
        images, binary masks and weight maps. 
    ----------
    Arguments
    ----------
            images : Array - list of input images
            masks : Array - list of output binary masks
            weights : Array - list of weight maps
            num_patches : Float - number of patches - Default: 25 for a 256x256 pixels input images
            shape : Array -  2D array with the size of the patches - Default: 256x256 pixels.
            bck_th : Float - Mean pixel intensity threshold for discarding patches containing mostly background. Default: 0.05
    ----------
    Returns
    ----------
            input_patches :  Array - list of input patches
            output_patches : Array - list of output binary masks patches
            weight_patches : Array - list of weight map patches.
    '''
    # Retrieve input image size 
    original_size = images[0].shape
    # Create empty lists to store patches 
    input_patches = []
    output_patches = []
    weight_patches = []
    rm_patches = []
    # Generate and store patches 
    for n in range( 0, len(images) ):
        image = images[n]
        mask = masks[n]
        weight = weights[n]
        for i in range( num_patches ):
          r = np.random.randint(0,original_size[0]-shape[0])
          c = np.random.randint(0,original_size[1]-shape[1])
          im_patch = image[ r : r + shape[0], c : c + shape[0] ]
          mask_patch = mask[ r : r + shape[0], c : c + shape[0] ]
          weight_patch = weight[ r : r + shape[0], c : c + shape[0] ]
          # Test to remove patches containing mostly background on the basis of the mean pixel intensity of the mask patch.
          if np.mean(img_as_float(mask_patch))> bck_th:
            input_patches.append( im_patch )
            output_patches.append( mask_patch )
            weight_patches.append( weight_patch )
          else:
            rm_patches.append(mask_patch)
    print('There are {} patches available for training.'.format(len(input_patches)))
    print('There are {} black patches which have been removed.'.format(len(rm_patches)))
    return (input_patches, output_patches, weight_patches)

def load_patches(path, folder, file_ext, w_map=False, re_size=False, new_shape=None):
    ''' Load image, binary mask or weight map patches. 
    ----------
    Arguments
    ----------
            path : String - Path to where the files are stored on drive
            folder : String - Folder in which the files are stored
            file_ext : String - File extension (e.g. '.tif')
            w_map : Boolean - Nature of the patches: weight map or image 
            re_size : Boolean - Resize or not loaded patches 
            new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches.

    ----------
    Returns
    ----------
            patches :  Array - List of loaded patches.
    '''
    filenames = [i for i in os.listdir( path + folder ) if i.endswith(file_ext)] # Load filenames
    filenames.sort() # Sort filenames
    print('Patches: ' + str(filenames))
    print( 'Number of patches available to train the model(s):' + str( len(filenames)) )
    if re_size is False:
        if w_map is False: 
            patches = [ img_as_ubyte( io.imread( path + folder + i ) ) for i in filenames ] # Create array storing input or output patches
        else:
            patches = [ io.imread( path + folder + i ) for i in filenames ] # Create array storing weight maps
    else:
        if w_map is False: 
            patches = [resize( img_as_ubyte( io.imread( path + folder + i ) ), output_shape=new_shape, anti_aliasing=True ) for i in filenames ] 
        else:
            patches = [resize( io.imread( path + folder + i ), output_shape=new_shape, anti_aliasing=True ) for i in filenames ]   
    return (patches)
    
def data_set_prep(image_patches, mask_patches, weight_patches, backbone_prep=None, stack_mode=False, verbose_mode=True):
    """ Normalises image intensity on [0,1] and transform them into 
        tensors of shape (width x height x 1).
    ----------
    Arguments
    ----------
			  image_patches : Array - list of input image patches 
			  mask_patches : Array - list of output binary mask patches
			  weight_patches : Array - list of weight map patches
			  stack_mode : Boolean - If True will concatenate W_t to Y_t
			  backbone_prep : String - Name of the encoder backbone.
    ---------
    Returns
    ---------
			 X_t : Tensor - list of input image patches
			 Y_t : Tensor - list of output binary mask patches
			 W_t : Tensor - list of output weight map patches.
    """
    # Compute input shape of tensors 
    patch_shape = image_patches[0].shape
    train_width = patch_shape[0]
    train_height = patch_shape[1]
    tensor_shape = ( train_width, train_height, 1 )
    if backbone_prep is None:
      # Image patches
      X_t = [x/np.max(x) for x in image_patches] # Normalize intensity between 0 and 1
      X_t = [np.reshape(x, tensor_shape ) for x in X_t] # Reshape images into tensor of input shape
      X_t = np.asarray(X_t, dtype=np.float32) # Give the right numerical format to tensors 
      # Weight map patches
      W_t = [x for x in weight_patches] 
      W_t = [np.reshape( x, tensor_shape ) for x in W_t]
      W_t = np.asarray(W_t)
      # Mask patches 
      Y_t = [x/np.max(x) for x in mask_patches]
      Y_t = [np.reshape( x, tensor_shape ) for x in Y_t]
      Y_t = np.asarray(Y_t, dtype=int)
      if stack_mode is True : 
        Y_t = np.concatenate((Y_t, W_t), axis=-1) # Concatenate together ground truth masks and weight maps
      else:
        pass 
    else:
      # Define image preprocessing function
      preprocess_input = get_preprocessing(backbone_prep)
      # Image patches
      X_t = [preprocess_input(x) for x in image_patches] # Normalize intensity between 0 and 1
      X_t = [np.reshape(x, tensor_shape ) for x in X_t] # Reshape images into tensor of input shape
      X_t = np.asarray(X_t, dtype=np.float32) # Give the right numerical format to tensors 
      # Weight map patches
      W_t = [x for x in weight_patches] 
      W_t = [np.reshape( x, tensor_shape ) for x in W_t]
      W_t = np.asarray(W_t)
      # Mask patches 
      Y_t = [x/np.max(x) for x in mask_patches]
      Y_t = [np.reshape( x, tensor_shape ) for x in Y_t]
      Y_t = np.asarray(Y_t, dtype=int)
      if stack_mode is True : 
        Y_t = np.concatenate((Y_t, W_t), axis=-1) # Concatenate together ground truth masks and weight maps
      else:
        pass
    if verbose_mode == True :
        print('There are {} tensors available.'.format(len(X_t)))
    else:
        pass
    return (X_t, Y_t, W_t)

def data_set_split( X_t, Y_t, W_t, train_ratio, val_ratio, test_ratio, seed=None, shuffle=True, save_data=False, train_path=None, test_path=None, train_folder=None, val_folder=None, test_folder=None):
    """Split the initial data set into training, validation and test data sets,
       and save these 3 data sets. 
    ----------
    Arguments
    ----------
              X_t : Tensor - list of input images
              Y_t : Tensor - list of output masks
              Y_t : Tensor - list of pixel-wise weight maps 
              save_data : Boolean - To save or not generated data sets
              path : String - Path to where to save generated data sets
              train_folder : String - Name of the folder for saving the training data set 
              val_folder : String - Name of the folder for saving the validation data set 
              test_folder : String - Name of the folder for saving the test data set .
    ----------
    Returns
    ----------
              X_train : Tensor - list of training input images
              Y_train : Tensor - list of training output masks
              W_train : Tensor - list of training pixel-wise weight maps
              X_test  : Tensor - list of test input images
              Y_test  : Tensor - list of test output masks
              W_test  : Tensor - list of test pixel-wise weight maps
              X_val   : Tensor - list of validation input images
              Y_val   : Tensor - list of validation output masks
              W_val   : Tensor - list of validation pixel-wise weight maps.
    """
    # Generate training, validation and test data sets
    X_train, X_test, Y_train, Y_test, W_train, W_test = train_test_split(X_t, Y_t, W_t, test_size = (1 - train_ratio), random_state = seed, shuffle = True)
    X_val, X_test, Y_val, Y_test, W_val, W_test = train_test_split(X_test, Y_test, W_test, test_size = test_ratio/(test_ratio + val_ratio), random_state = seed, shuffle = False) 
    print('Training data set: '+ str(len(X_train)) +' images | Validation data set: '+ str(len(X_val)) + ' images | Test data set: '+ str(len(X_test)) + ' images.')
    # Save training patches as 32-bits.tif files
    if save_data == True:
      for i in range(0,len(X_train)):
        io.imsave(train_path + train_folder + 'Training_Images/' + 'Training_Image_'+ str(i)+'.tif', X_train[i], plugin='tifffile',check_contrast=False)
        io.imsave(train_path + train_folder + 'Training_Masks/' + 'Training_Mask_'+ str(i)+'.tif', Y_train[i], plugin='tifffile',check_contrast=False)
        io.imsave(train_path + train_folder +'Training_Weights/' + 'Training_Weight_'+ str(i)+'.tif', W_train[i], plugin='tifffile',check_contrast=False)
      print('Training data set has been saved.')
      # Save validation patches as 32-bits.tif files
      for i in range(0,len(X_val)):
        io.imsave(train_path + val_folder +  'Validation_Images/' + 'Validation_Image_'+ str(i)+'.tif', X_val[i], plugin='tifffile',check_contrast=False)
        io.imsave(train_path + val_folder + 'Validation_Masks/' + 'Validation_Mask_'+ str(i)+'.tif', Y_val[i], plugin='tifffile',check_contrast=False)
        io.imsave(train_path + val_folder + 'Validation_Weights/' + 'Validation_Weight_'+ str(i)+'.tif', W_val[i], plugin='tifffile',check_contrast=False)
      print('Validation data set has been saved.')
      # Save test patches as 32-bits.tif files
      for i in range(0,len(X_test)):
        io.imsave(test_path + test_folder + 'Test_Images/' + 'Test_Image_'+ str(i)+'.tif', X_test[i], plugin='tifffile',check_contrast=False)
        io.imsave(test_path + test_folder + 'Test_Masks/' + 'Test_Mask_'+ str(i)+'.tif', Y_test[i], plugin='tifffile',check_contrast=False)
        io.imsave(test_path + test_folder + 'Test_Weights/' + 'Test_Weight_'+ str(i)+'.tif', W_test[i], plugin='tifffile',check_contrast=False)
      print('Test data set has been saved.')
    else:
      pass
    return (X_train, Y_train, W_train, X_test, Y_test, W_test, X_val, Y_val, W_val)

class DataGenerator(Sequence):
    """Class for creating custom image augmentation generators compatible with Albumentations & Tensorflow/Keras.
    ----------
    Arguments
    ----------
              X : Tensors - Images
              M : Tensors - Masks
              aug_scheme : Albumentation object - Augmentation scheme
              dim: Tuple - Images / Mask  2D dimensions e.g. (256,256)
              batch_size: Integer - Size of the batches to be generated e.g. 32
              shuffle : Boolean - Shuffle images/ masks order from one batch to the other.
    ----------
    Returns
    ----------
              XX : Function - Augemented image generator 
              YY : Function - Augmented mask generator 
    """
    def __init__(self, X, M, aug_scheme, dim, batch_size=16, shuffle=True):
        'Initialization'
        self.X = X
        self.M = M
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.aug = aug_scheme
        self.dim = dim
        self.on_epoch_end()
    
    def __len__(self):
        'Compute the number of batches per epoch to be generated'
        return (int(np.floor((len(self.X) / self.batch_size) / 1) ))
        
    def __getitem__(self, index):
        'Generate a single batch of data'
        # Generate indexes of the batch
        end_index = min((index+1)*self.batch_size, len(self.indexes))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, Y = self.__data_generation(indexes)
        return ( X, Y )

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.X))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # X : (n_samples, *dim, n_channels)        
        batch_size = len(indexes)
        # Initialization
        XX = np.empty((batch_size, self.dim[1], self.dim[0], 1), dtype='float32')
        YY = np.empty((batch_size, self.dim[1], self.dim[0], 2), dtype='float32')
        # Generate data
        for i, ID in enumerate(indexes):
            # Store sample
            img = self.X[ID]
            mask = self.M[ID]
            # Store class
            augmented = self.aug(image=img, mask=mask)
            aug_img = augmented['image']
            aug_mask = augmented['mask']
            YY[i,] = aug_mask.astype('float32')
            XX[i,] = aug_img.astype('float32')
        return ( XX, YY )

###### Validation metrics functions ######  

def confusion_matrix_elements(y_true, y_pred):
    """ Compute confusion matrix elements.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              (TP, TN, FP, FN): Floats - Confusion matrix elements.
    """
    # Convert y_true to float32 to match y_pred
    y_true = tf.cast(y_true, tf.float32)
    # Positive & negative samples in training data set
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = (1 - y_pos)
    # Positive & negative samples in predicted data set
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg =(1 - y_pred_pos)
    # Elements of the matrix 
    tp = K.sum(y_pos * y_pred_pos) # True positives
    tn = K.sum(y_neg * y_pred_neg) # True negatives
    fp = K.sum(y_neg * y_pred_pos) # False positives
    fn = K.sum(y_pos * y_pred_neg) # False neagtives
    return (tp, tn, fp, fn)

def accuracy(y_true, y_pred):
    """ Accuracy on classification 
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              A : Tensor -  Precision.  
    """
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)   
    A = (tp + tn) / (tp + tn + fp + fn + K.epsilon())
    return ( A )

def error(y_true, y_pred):
    """ Error on classification
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              A : Tensor -  Precision.  
    """
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)   
    E = (fp + fn) / (tp + tn + fp + fn + K.epsilon())
    return ( E )

def precision(y_true, y_pred):
    """ Precision or True positive Rate 
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              P : Tensor -  Precision.  
    """
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)   
    P = tp / (tp + fp + K.epsilon())
    return ( P )

def recall(y_true, y_pred):
    """ Recall or Sensitivity or True Positive Rate 
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              R: Tensor - Recall.
    """  
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)
    R = tp / (tp + fn + K.epsilon())
    return ( R )

def F1_score(y_true, y_pred):
    """F1 score or Dice coefficient
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              F1_s : Tensor - F1 score or Dice coefficient.
    """  
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    F1_s = 2 * ((prec * rec)/(prec + rec + K.epsilon()))
    return ( F1_s )

def specificity(y_true, y_pred):
    """ Specificity of True Negative Rate.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              R: Tensor - Recall.
    """  
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)
    S = tn / (tn + fp + K.epsilon())
    return ( S )
    
def fpr(y_true, y_pred):
    """ False positive rate = 1-Specificity 
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              R: Tensor - Recall.
    """  
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)
    F_p = fp / (tn + fp + K.epsilon())
    return (F_p)

def IoU(y_true, y_pred):
    """Jaccard index or mean IoU 
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              J: Tensor - Jaccard index or mean IoU.
    """ 
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)
    J =  tp / (tp+ fn + fp + K.epsilon())
    return ( J )

def Matthews_correl(y_true, y_pred):
    """
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              M_correl : Tensor - Matthews correlation coefficient.
    """ 
    tp, tn, fp, fn = confusion_matrix_elements(y_true, y_pred)
    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    M_correl = numerator / (denominator + K.epsilon())
    return ( M_correl )

def auroc(y_true, y_pred):
    """
    ----------
    Arguments
    ----------
               y_true : Tensor - Ground binary mask.
               y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
               auc: Tensor - Area Under ROC Curve (AUROC).
    """ 
    auc = tf.metrics.auc(y_true, y_pred)[1]
    tf.compat.v1.keras.backend.get_session().run(tf.local_variables_initializer())
    return (auc) 

def us_accuracy_bis(y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes binary accuracy.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ---------
    Returns
    ---------
              unstack_bin_acc : Tensor - Binary prediction accuracy.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_bin_acc = binary_accuracy(seg,y_pred)
    return (u_bin_acc)
    
def us_accuracy(y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes accuracy.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ---------
    Returns
    ---------
              unstack_bin_acc : Tensor - Binary prediction accuracy.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_acc = accuracy(seg,y_pred)
    return (u_acc)

def us_error(y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes error.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ---------
    Returns
    ---------
              unstack_bin_acc : Tensor - Binary prediction accuracy.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_err = error(seg,y_pred)
    return (u_err)
    
def us_IoU(y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes mean IoU.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_IoU : Tensor - Prediction mean IoU.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_IoU = IoU(seg,y_pred)
    return (u_IoU)
    
def us_precision (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes precision.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_precision : Tensor - Prediction precision.

    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_prec = precision(seg, y_pred)
    return (u_prec)
    
def us_recall (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes recall.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Prediction recall.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_rec = recall(seg, y_pred)
    return (u_rec) 
    
def us_F1_score (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes F1 score (Dice coefficient).
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Prediction F1 score.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_F1_sc = F1_score(seg, y_pred)
    return (u_F1_sc)

def us_specificity (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes specificity.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Prediction recall.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_spec = specificity(seg, y_pred)
    return (u_spec)

def us_fpr (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes false positive rate.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Prediction recall.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_f = fpr(seg, y_pred)
    return (u_f) 

def us_Matthews_correl (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes Matthews correlation coefficient.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Matthews correlation coefficient.

    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass    
    
    u_M_cor = Matthews_correl(seg, y_pred)
    return (u_M_cor )

def us_auroc (y_true, y_pred):
    """ Unstacks the mask from the weights in the output tensor for
        segmentation and computes the Area under ROC curve (AUROC).
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground binary mask + weight maps.
              y_pred : Tensor - Predicted segmentation mask.
    ----------
    Returns
    ----------
              unstack_recall : Tensor - Prediction Area under ROC curve.
    """
    try:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)
        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass
    
    u_auc = auroc (seg, y_pred)
    return (u_auc) 

###### Loss functions ###### 

def wbce_loss(beta):
	""" Weighted binary cross-entropy loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
			  beta : Float - Weight coefficient used on positive examples. 
			  / If beta > 1 decreases false negatives and increase recall./
			  / If beta < 1 decreases false positives and increases precision./
    ----------
    Returns
    ----------
              loss : Tensorflow function - Loss function.
    """	
	def loss(y_true, y_pred):
		weight_a = beta * tf.cast(y_true, tf.float32)
		weight_b = 1 - tf.cast(y_true, tf.float32)
		o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
		return (tf.reduce_mean(o))
	
	return (loss)
  
def bbce_loss(beta):
    """ Balanced binary cross-entropy loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
			  beta : Float - Weight coefficient used on positive and negative examples.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        loss = tf.reduce_mean(o)
    return (loss)

def focal_loss(alpha=0.25, gamma=2):
    """ Focal loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
			  alpha : Float - First focal loss coefficient.
			  gamma : Float - Second focal loss coefficient.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)
    return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits) * (weight_a + weight_b) + (logits * weight_b)) 
    
    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        l = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)
        loss =tf.reduce_mean(l)
    return (loss)

def dice_loss(y_true, y_pred):
    """ Dice loss function.
    ----------
    Parameters
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.math.sigmoid(y_pred)
    numerator = 2 * tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    loss = 1 - (numerator / denominator)
    return (loss)

def tversky_loss(beta):
    """ Tversky loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
			  beta : Float - Weight coefficient used on positive and negative examples.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = y_true * y_pred
        denominator = y_true * y_pred + beta * (1 - y_true) * y_pred + (1 - beta) * y_true * (1 - y_pred)
        return (1 - tf.reduce_sum(numerator) / tf.reduce_sum(denominator))
    return (loss)
    
def bce_loss(y_true, y_pred):
    """ Binary cross-entropy loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    loss = binary_crossentropy(y_true, y_pred) 
    return (loss)

def bce_dice_loss(y_true, y_pred):
    """ Combined balanced binary cross-entropy and Dice loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
			  y_pred : Tensor - Predicted segmentation masks.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return (1 - numerator / denominator)
        
    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    loss = tf.reduce_mean(o)
    return (loss)
    
def pixelwise_wbce(y_true, y_pred):
    """ Pixel-wise weighted binary cross-entropy loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
              y_pred : Tensor - Predicted segmentation masks.
    ----------
    Returns
    ----------
			  loss : loss : Tensorflow function - Loss function.
    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)

        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass

    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)

    loss = K.mean(math_ops.multiply(weight, entropy), axis=-1)
    return (loss)
 
def pixelwise_wbce_dice(y_true, y_pred):
    """ Combined pixel-wise weighted binary cross-entropy and dice coefficient loss function.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Stack of groundtruth segmentation masks + weight maps.
              y_pred : Tensor - Predicted segmentation masks.
    ----------
    Returns
    ----------
			  loss : Tensorflow function - Loss function.
    """
    try:
        # The weights are passed as part of the y_true tensor:
        [seg, weight] = tf.unstack(y_true, 2, axis=-1)
        seg = tf.expand_dims(seg, -1)
        weight = tf.expand_dims(weight, -1)
    except:
        pass
    
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred) + K.epsilon()
        return (1 - numerator / denominator)
        
    epsilon = tf.convert_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_pred = tf.math.log(y_pred / (1 - y_pred))

    zeros = array_ops.zeros_like(y_pred, dtype=y_pred.dtype)
    cond = (y_pred >= zeros)
    relu_logits = math_ops.select(cond, y_pred, zeros)
    neg_abs_logits = math_ops.select(cond, -y_pred, y_pred)
    entropy = math_ops.add(relu_logits - y_pred * seg, math_ops.log1p(math_ops.exp(neg_abs_logits)), name=None)

    pwbce = K.mean(math_ops.multiply(weight, entropy), axis=-1)
    
    o = pwbce + dice_loss(seg, y_pred) 
    
    loss = tf.reduce_mean(o)
    return (loss)
    
###### Models ###### 

def UNet_LW(train_width, train_height):
    """Light-weight U-Net model.
    ---------------
    Fixed features
    ---------------
              Number of resolution levels: 3
              Initial feature maps number: 16
              Convolution kernel: 3x3 pixels
              Activation function: elu 
              Kernel initializer: 'he_normal'(Truncated normal distribution centered on 0 of fixed std) 
              Dropout: Yes - 0.1
              Pooling: Average pooling - 2x in both horizontal and vertical direction.
              Padding: Yes - 'same'.
    ----------
    Arguments
    ----------
			  train_width : Int - Width of the input training image tensor.
              train_height : Int - Height of the input training image tensor.
    ----------
    Returns
    ----------
	          model : Tensorflow model - Ready to compile Light-weight U-Net model.
    """
    inputs = Input((train_width, train_height, 1))

    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
    c1 = Dropout(0.1) (c1)
    c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
    p1 = AveragePooling2D((2, 2)) (c1)

    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
    c2 = Dropout(0.1) (c2)
    c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
    p2 = AveragePooling2D((2, 2)) (c2)

    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
    c3 = Dropout(0.2) (c3)
    c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
    p3 = AveragePooling2D((2, 2)) (c3)

    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
    c4 = Dropout(0.2) (c4)
    c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)

    u5 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c4)
    u5 = concatenate([u5, c3])
    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u5)
    c5 = Dropout(0.2) (c5)
    c5 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

    u6 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c2])
    c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
    c6 = Dropout(0.1) (c6)
    c6 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

    u7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c1], axis=3)
    c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
    c7 = Dropout(0.1) (c7)
    c7 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

    outputs = Conv2D(1, (1, 1), activation='sigmoid') (c7)

    model = Model(inputs=[inputs], outputs=[outputs])

    return (model)
    
def UNet_OR(train_width, train_height):
    """Original U-Net model.
    ---------------
    Fixed features
    ---------------
              Number of resolution levels: 4
              Initial feature maps number: 64
              Convolution kernel: 2x2 pixels
              Activation function: Relu 
              Kernel initializer: 'he_normal'(Truncated normal distribution centered on 0 of fixed std) 
              Dropout: Yes - 0.2
              Pooling: Max pooling - 2x in both horizontal and vertical direction.
              Padding: Yes - 'same'.
    ----------
    ----------
    Arguments
    ----------
			  train_width : Int - Width of the input training image tensor.
              train_height : Int - Height of the input training image tensor.
    ----------
    Returns
    ----------
	          model : Tensorflow model - Ready to compile original U-Net model.
    """
    inputs =  Input((train_width, train_height, 1))
	
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    c1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)
    
    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p1)
    c2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)
    
    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p2)
    c3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c3)
    p3 = MaxPooling2D(pool_size=(2, 2))(c3)
    
    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p3)
    c4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c4)
    d4 = Dropout(0.5)(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(d4)

    c5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(p4)
    c5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c5)
    d5 = Dropout(0.5)(c5)

    u6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(d5))
    m6 = Concatenate(axis = 3)([d4,u6]) 
    c6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m6)
    c6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c6)

    u7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c6))
    m7 = Concatenate(axis = 3)([c3,u7])
    c7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m7)
    c7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c7)

    u8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c7))
    m8 = Concatenate(axis = 3)([c2,u8])
    c8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m8)
    c8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c8)

    u9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(c8))
    m9 = Concatenate(axis = 3)([c1,u9])
    c9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(m9)
    c9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(c9)
    
    outputs = Conv2D(1, 1, activation='sigmoid') (c9)

    model = Model(inputs=[inputs], outputs=[outputs])

    return (model)

###### Plots ######

def example_images_plot(images, names, cmaps, contour=False, contour_image=None, contour_target=None, color_bar=False, color_bar_target=None, save_plot=False, path=None, file_name=None):
    """ Generic function displaying example images as contiguous panels in a 
        single row plot. Ex: display example image / binary mask / weight map.
    ----------
    Arguments
    ----------
              images : List - Image files to be plotted
              names : List - Names of image files to be plotted
              cmaps : List - Names of colormaps used for plotting
              contour : Boolean - Compute contours of objects or not
              contour_image : Array - Image file on which objects contour are computed
              contour_target : String - Name of the image file on which the contour is to be plotted
              color_bar : Boolean - Display colorbar or not
              color_bar_target : String -  Name of the image file on which the colorbar is to be plotted
              save_plot : Boolean - Save plot or not
              path : String - Path to save file on drive
              file_name : String - File name.
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Number of image panels
    N_p=len(names) 
    # Plot size 
    if color_bar is True: 
      p_size=((4*N_p),3.75)
    else: 
      p_size=((4*N_p),4.0)
    # Plot figure
    fig,axes = plt.subplots(1, N_p, figsize=p_size)
    for i in range(0,N_p):
      axes[i].plot
      im=axes[i].imshow(images[i], cmaps[i], origin = 'lower')
      if contour is True:
        contours = find_contours(contour_image, 0.5) # Compute contours of foreground instances in a binary mask
        if names[i] == contour_target:
          for n, contour in enumerate(contours):
            axes[i].plot(contour[:, 1], contour[:, 0], linewidth = 1.00 , color ='red')
        else:
          pass
      else:
        pass
      axes[i].set_title(names[i], fontsize = 18)
      axes[i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      if color_bar is True and names[i] == color_bar_target:
          cbar=fig.colorbar(im,ax=axes[i],cax=make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%"))
      else:
        pass
    fig.tight_layout()
    if save_plot is True: 
        fig.savefig(path + file_name +'example_images.pdf', dpi=300) # Save the plot 
        fig.savefig(path + file_name +'example_images.png', dpi=300) 
    else:
      pass
    return

def visual_augmentations(image, mask, original_image=None, original_mask=None, c_map='gray', fontsize = 18):
    """ Generic function displaying example image /mask and their augmented counterparts for a given augmentation scheme.
    ----------
    Arguments
    ----------
              image : Array - Augmented image file to be plotted
              mask : Array - Augmented mask file to be plotted
              original_image : Array - Original image file to be plotted
              original_mask : Array - Original mask file to be plotted
              c_map : String - Names of the colormap used for plotting.
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    if original_image is None and original_mask is None:
        fig, ax = plt.subplots(2, 1, figsize=(8, 8))
        ax[0].imshow(image, c_map, origin = 'lower')
        ax[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1].imshow(mask, c_map,  origin = 'lower')
        ax[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        fig.tight_layout()
    else:
        fig, ax = plt.subplots(2, 2, figsize=(8, 8))
        ax[0,0].imshow(original_image, c_map, origin = 'lower')
        ax[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[0,0].set_title('Original image', fontsize=fontsize)
        ax[1,0].imshow(original_mask, c_map, origin = 'lower')
        ax[1,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1,0].set_title('Original mask', fontsize=fontsize)
        ax[0,1].imshow(image, c_map, origin = 'lower')
        ax[0,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[0,1].set_title('Transformed image', fontsize=fontsize)
        ax[1,1].imshow(mask, c_map, origin = 'lower')
        ax[1,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        ax[1,1].set_title('Transformed mask', fontsize=fontsize)
        fig.tight_layout()
    return

def example_augmentations(data_generator, N_im, cmaps=['gray','gray','viridis'], save_plot=False, path=None, file_name=None):
    """ Generic function displaying several example image /mask and their augmented counterparts for a given augmentation scheme.
    ----------
    Arguments
    ----------
              data_generator : Object - Data generator
              N_im : Integer - Number of example images / masks / weight maps to be ploted
              cmaps : List - Names of colormaps used for plotting
              save_plot : Boolean - Save plot or not
              path : String - Path to save file on drive
              file_name : String - File name.
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Plot sample images 
    fig,axes = plt.subplots(3,N_im,figsize=((N_im*3.3),10.0))
    for X_b, Y_b in data_generator:
        for i in range (0,N_im):
            axes[0,i].plot
            axes[0,i].imshow(X_b[i][:,:,0],  cmap = cmaps[0], origin = 'lower') # Image patch 
            axes[0,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            axes[1,i].plot
            axes[1,i].imshow(Y_b[i][:,:,0],  cmap = cmaps[1], origin = 'lower') # Mask patch 
            axes[1,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
            axes[2,i].plot
            axes[2,i].imshow(Y_b[i][:,:,1],  cmap = cmaps[2], origin = 'lower') # Weight patch
            contours = find_contours(Y_b[i][:,:,0], 0.5) # Compute contours of foreground instances in a binary mask
            for n, contour in enumerate(contours):
                axes[2,i].plot(contour[:, 1], contour[:, 0], linewidth = 1.00 , color ='red')
            axes[2,i].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        break
    fig.tight_layout()
    fig.show()
    if save_plot is True: 
        fig.savefig(path + file_name +'example_images.pdf', dpi=300) # Save the plot 
        #fig.savefig(path + file_name +'example_images.png', dpi=300) 
    else:
      pass
    return
    
def training_history_plot(training_history, metrics_train, metrics_valid, names, save_plot=False, path=None, file_name=None):
    """ Function generating grouped plots of training metrics recorded during
        model training.
    ----------
    Arguments
    ----------
              training_history : Tensorflow object - Metrics data  
              metrics_train : List - Metrics recorded during training
              metrics_valid : List - Metrics recorded during training
              names : List - Name of metrics 
              save_plot : Boolean - Save plot or not
              path : String - Path to save file on drive
              file_name : String - File name.
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Number of image panels
    N_p=len(metrics_train)
    # Plot Size 
    p_size=((4*N_p),4.0)
    fig,axes = plt.subplots(1,N_p,figsize=p_size)
    for i in range(0,N_p):
      if metrics_train[i] not in ('loss','us_specificity', 'specificity','us_fpr','fpr'):
        y_max = 1.04
      else:
        y_max = np.max(training_history.history[metrics_valid[i]])+0.007
      if metrics_train[i] in ('us_specificity','specificity','us_fpr','fpr'):
        y_min = np.min(training_history.history[metrics_valid[i]])-0.007
      else:
        y_min = np.min(training_history.history[metrics_valid[i]])-0.07
      axes[i].plot
      axes[i].plot(training_history.history[metrics_train[i]], color='C0', linestyle='-', linewidth=2.0, alpha=0.85)
      axes[i].plot(training_history.history[metrics_valid[i]], color='C3', linestyle='-', linewidth=2.0, alpha=0.85)
      axes[i].legend(['training' , 'validation' ],fontsize=14)
      axes[i].set_ylim([y_min,y_max])
      axes[i].set_ylabel(names[i],fontsize=16)
      axes[i].set_xlabel('Epoch',fontsize=16)
      axes[i].tick_params(direction='in', axis='both', which='both', labelsize=14)
    fig.tight_layout()
    if save_plot is True: 
      fig.savefig( path + 'Training_Hist_' + file_name + '.pdf', dpi=300)
      # fig.savefig( path + 'Training_Hist_' + file_name + '.png', dpi=300) 
    else:
      pass
    return

##################################################### Evaluation ###################################################

###### Semantic segmentation ###### 

def import_trained_models(l_models, path, l_rate=None, loss_func=None):
    """ Load trained DL models for prediction on unseen images or fine tuning. 
    ----------
    Arguments
    ----------
              l_models : Array - List of models to be loaded
              path : String - Path to where the files are stored on drive
              l_rate : Float - Learning rate if compilation is needed
              loss_func : String - Loss function if compilation is needed.
    ----------
    Returns
    ----------
              trained_models: Tensorflow object - Trained tensorflow model ready for inference .
    """
    trained_models = []
    if l_rate is None and loss_func is None: 
      for i in range (0, len(l_models)):
        model = load_model(path + l_models[i] +'.h5',compile=False)
        trained_models.append(model); # Import model for prediction only 
    else: 
      for i in range (0, len(l_models)):
        model = load_model(path + l_models[i] +'.h5',compile=False)
        model.compile(optimizer = Adam(learning_rate = l_rate), loss = loss_func)
        trained_models.append(model); # Import compiled models (prediction and fine tuning) 
    print('There are {} trained models ready for evaluation.'.format(len(trained_models)))
    return (trained_models)

def probability_maps(models_l, backbones_l, image_patches, mask_patches, weight_patches):
      """ Predict segmentation probability maps for an arbitrary number of trained 
          models on all the images of a given input image set.
      ----------
      Arguments
      ----------
                models_l : Array - List of model imported trained models
                backbones_l : Array - List of backbones used by the imported trained models
                image_patches : Array - list of input image patches 
                mask_patches : Array - list of output binary mask patches
                weight_patches : Array - list of weight map patches.
      ----------
      Returns
      ----------
                p_maps : Array - Probability maps of the input image set for each input trained model. 
      """
      p_maps = []
      for i in range (0, len(models_l)):
          X_T, Y_T, W_T = data_set_prep(image_patches, mask_patches, weight_patches, backbone_prep=backbones_l[i], stack_mode=False, verbose_mode=False)
          prob = models_l[i].predict(X_T)
          p_maps.append(prob)
      print('Probability maps have been computed for {} models.'.format(len(models_l)))
      return (p_maps)

def precision_recall_curve_data(y_true, y_pred):
    """ Compute precision-recall curves for various thresholds for an image set 
        and determines the optimal threshold for probability maps binarization.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation masks
              y_pred : Tensor - Predicted probability maps.
    ----------
    Returns
    ----------
              precision : Array - Precision values at various thresholds
              recall :  Array - Recall values at various thresholds
              threshold : Array - Thresholds used for binarisation 
              F1_score : Array - F1 score values at various thresholds
              ix : Integer - Index of the best binarisation threshold
              B_T : Float - Best binarisation threshold
              F1_T : Float - F1 score at best binarisation threshold.
    """
    l_t = []
    l_p = []
    for i in range (0,y_true.shape[0]):
      l_t.append(y_true[i][:,:,0].astype(int).flatten())
      l_p.append(y_pred[i][:,:,0].flatten())
    # Concatenate data for all images in the data set
    data_true = np.concatenate(l_t)
    data_pred = np.concatenate(l_p)
    # Compute precision and recall at different thresholds and store results
    precision, recall, thresholds = precision_recall_curve(data_true, data_pred, pos_label=1)
    # Compute F1 score
    F1_score = (2 * precision * recall) / (precision + recall + 1e-9)
    # Locate the threshold value of the largest F1 score
    ix = np.argmax(F1_score)
    B_T = thresholds[ix]
    F1_T = F1_score[ix]
    return (precision, recall, thresholds, F1_score, ix, B_T, F1_T)

def seg_threshold(y_true, y_pred, t_models):
  """ Compute precision-recall curves for various thresholds for an image set 
      and determines the optimal threshold for probability maps binarization.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation masks
              y_pred : Tensor - Predicted probability maps
              t_models : Array - List of compiled tensorflow models.
    ----------
    Returns
    ----------
              BT_L : Array - Values of the best binarisation threshold for each model
              F1T_L : Array - Values of F1 score at the best binarisation threshold for each model.
  """
  BT_L = []
  F1T_L = []
  for i in range (0, len(t_models)):
    precision, recall, thresholds, F1_score, ix, B_T, F1_T = precision_recall_curve_data(y_true, y_pred[i]) 
    BT_L.append(B_T)
    F1T_L.append(F1_T)
  print('Binarisation thresholds have been computed for {} models.'.format(len(t_models)))
  return (BT_L,F1T_L)

def performance_metrics(y_true, y_pred, threshold, average):
    """ Compute various segmentation performance metrics: accuracy, precision, 
        recall, F1 score, IoU, Matthews correlation coefficient, AUC of ROC.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation masks
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              average : String - Determines the averaging performed on the data ('micro': per indivual label, 
                        'macro': per class, 'weighted': per class corrected for class imbalance)
    ----------
    Returns
    ----------
              perf_metrics : Array - Values of the performance metrics for each image of the data set
              perf_metrics_mean : Array - Average values of the performance metrics over the whole data set
              perf_metrics_std : Array - Std values of the performance metrics over the whole data set.
    """
    # Create empty lists to store performance metrics
    l_acc = []
    l_prec = []
    l_rec = []
    l_f1 = []
    l_iou = []
    l_mcc = []
    l_roc_auc = []
    for i in range (0, len(y_true)):
      y_t = y_true[i][:,:,0].astype(int).flatten()
      y_p = (y_pred[i][:,:,0]>threshold).astype(int).flatten()
      # Compute and store performance metrics 
      acc = accuracy_score(y_t, y_p) # Accuracy
      l_acc.append(acc)
      prec = precision_score(y_t, y_p, average = average) # Precision
      l_prec.append(prec)
      rec = recall_score(y_t, y_p, average = average) # Recall
      l_rec.append(rec)
      f1 = f1_score(y_t, y_p, average = average) # F1 score 
      l_f1.append(f1)
      iou = jaccard_score(y_t, y_p, average = average) # IoU 
      l_iou.append(iou)
      mcc = matthews_corrcoef(y_t, y_p) # MCC 
      l_mcc.append(mcc)
      roc_auc = roc_auc_score(y_t, y_p, average = average) # AUC ROC
      l_roc_auc.append(roc_auc)
    # Create an array to store all the data together
    perf_metrics = np.zeros((len(y_true),7), dtype=float)
    perf_metrics[:,0]=l_acc
    perf_metrics[:,1]=l_prec
    perf_metrics[:,2]=l_rec
    perf_metrics[:,3]=l_f1
    perf_metrics[:,4]=l_iou
    perf_metrics[:,5]=l_mcc
    perf_metrics[:,6]=l_roc_auc
    # Compute mean for the different metrics 
    perf_mean = np.mean(perf_metrics, axis = 0)
    # Compute std for the different metrics 
    perf_std = np.std(perf_metrics, axis = 0)
    return (perf_metrics, perf_mean, perf_std)
    
def sem_perf_metrics_plot_df(metrics, metrics_names):
    """ Create a Pandas data frame object for further plotting 
        with Seaborn or other high level python plotting API.
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings.
    ----------
    Returns
    ----------
              df_metrics : Pandas DF - Dataframe of performance metrics over all image set.
    """
    # Create a data frame for each computed metrics 
    l_f = []
    for i in range(0, len(metrics_names)):
      l_f.append(pd.DataFrame(metrics[:,i]))
    l_fn = []
    # Assign a metrics name to each data frame 
    for i in range(0, len(metrics_names)):
      l_fn.append(l_f[i].assign(Metrics = metrics_names[i]))
    # Concatenate name dataframes in a single data frame 
    cdf = pd.concat(l_fn) 
    # Create final data frame   
    df_metrics = pd.melt(cdf, id_vars=['Metrics'])       
    return (df_metrics)

def sem_perf_metrics_df(metrics, metrics_names, model_name=None, path=None):
    """ Create and save a Pandas data frame object containing semantic segmentation 
        performance metrics for a given model data set. 
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics / Lines: Values for each image of the data set
              metrics_names : List - Name of the performance metrics given as strings
              model_name : String - Model name
              path : String - Path to save file on drive.
    ----------
    Returns
    ----------
              df_metrics : Pandas DF - Dataframe of performance metrics over all the image set.
    """
    df_sem_metrics = pd.DataFrame(metrics) # Create a data frame  
    df_sem_metrics.columns = metrics_names # Add a header  
    if path is not None:
        df_sem_metrics.to_csv(path +'Semantic_Segmentation_Performance_Metrics_'+ model_name +'.csv', sep = ",") # Save as csv file
    else:
        pass 
    return (df_sem_metrics) 

def sem_perf_metrics_stat_df(metrics, metrics_names, model_name=None, path=None):
    """ Create and save a Pandas data frame object containing mean +/- Std of 
        semantic segmentation performance metrics for a given model data set. 
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics / Lines: Values for each image of the data set
              metrics_names : List - Name of the performance metrics given as strings
              model_name : String - Model name
              path : String - Path to save file on drive.
    ----------
    Returns
    ----------
              df_av_metrics : Pandas DF - Dataframe of average of the performance metrics over all the image set.
              df_std_metrics : Pandas DF - Dataframe of Std of the performance metrics over all the image set.
    """
    # Mean
    df_av_metrics = pd.DataFrame(np.mean(metrics, axis = 0)) # Create a data frame  
    df_av_metrics.columns = metrics_names # Add a header  
    if path is not None:
        df_av_metrics.to_csv(path +'Semantic_Segmentation_Performance_Metrics_Mean_'+ model_name +'.csv', sep = ",") 
    else: 
        pass
    # Std 
    df_std_metrics = pd.DataFrame(np.std(metrics, axis = 0)) # Create a data frame  
    df_std_metrics.columns = metrics_names # Add a header  
    if path is not None:
        df_std_metrics.to_csv(path +'Semantic_Segmentation_Performance_Metrics_Std_'+ model_name +'.csv', sep = ",") 
    else:
        pass
    return (df_av_metrics, df_std_metrics)   
    
def sem_perf_metrics_table(metrics_l, metrics_names, model_names_l, latex_tab=None):
    """ Create and save a latex table object containing mean +/- Std of 
        semantic segmentation performance metrics for a given model data set. 
    ----------
    Arguments
    ----------
              metrics_l : List - List of arrays: Columns: Performance metrics / Lines: Values for each image of the data set
              metrics_names : List - Name of the performance metrics given as strings
              model_names_l : List - Model names
              path : String - Path to save file on drive
              latex_tab : Boolean  - To plot the results table formated in latex.
    ----------
    Returns
    ----------
              table_metrics : Latex object - Dataframe of table object containing mean +/- Std of performance metrics.

    """
    table_metrics = []
    # Create first row 
    f_r =['Model'] + metrics_names
    # Append first row 
    table_metrics.append(f_r)
    # Generate and append other rows
    for i in range(0, len(model_names_l)):
      o_r_1 = []
      o_r_1.append(model_names_l[i])
      mean_p = np.mean(metrics_l[i], axis = 0)
      std_p = np.std(metrics_l[i], axis = 0)
      o_r_2 = []
      for j in range (0, len(mean_p)):
        o_r_2.append(str(np.around(mean_p[j], decimals=2)) + r'$\pm$' + str(np.around(std_p[j], decimals=2)))
      o_r = o_r_1 + o_r_2
      table_metrics.append(o_r) 
    # Generate printable table
    table = Texttable()
    table.set_max_width(0)
    table.set_cols_align(["c"] * (len(metrics_names)+1))
    table.set_deco(Texttable.HEADER | Texttable.VLINES)
    table.add_rows(table_metrics)
    # Print table
    print(table.draw())
    # Print table with Latex formatting to export in an external latex doc
    if latex_tab is not None :
      print(latextable.draw_latex(table, caption=" Semantic segmentation performance metrics."))
    return (table_metrics)

def confusion_matrix_set(y_true, y_pred, threshold, im_set=True, norm=True):
    """ Compute the confusion matrix of an image or the average confusion
        matrix +/- std over an image set.
    ----------
    Arguments
    ----------
			  y_true : Tensor - Ground truth segmentation masks.
              y_pred : Tensor - Predicted probability map.
              threshold : Float - Threshold for binary segmentation of the probability map.
              im_set : Boolean - To choose between image set or single image.
              norm : Boolean - To choose between the raw or normalised confusion matrix.
    ----------
    Returns
    ----------
			  cm : Array - confusion matrix.
			  cm_mean : Array - average confusion matrix over an image set.
			  cm_std : Array - std of confusion matrix over an image set.
    """
    if im_set == False:
      y_t = y_true[:,:,0].astype(int).flatten()
      y_p = (y_pred[:,:,0]>threshold).astype(int).flatten()
      cm = confusion_matrix(y_t, y_p)
      if norm == True:
        cm = np.divide(cm.astype('float') , cm.sum(axis=1)[:, np.newaxis])
      else:
        pass
      return (cm)
    else:
      l_cm = []
      for i in range (0, len(y_true)):
        y_t = y_true[i][:,:,0].astype(int).flatten()
        y_p = (y_pred[i][:,:,0]>threshold).astype(int).flatten()
        cm_bis = confusion_matrix(y_t, y_p)
        if norm == True:
          cm_bis = np.divide(cm_bis.astype('float') , cm_bis.sum(axis=1)[:, np.newaxis])
        else:
          pass
        l_cm.append(cm_bis)
      cm_ar = np.asarray(np.stack(l_cm))
      cm_mean = np.mean(cm_ar, axis = 0)
      cm_std = np.std(cm_ar, axis = 0)
      return (cm_mean, cm_std)
    return 

def confusion_matrix_image(y_true, y_pred, threshold, color_scheme='CBMY'):
    """ Generate a color-coded image of the confusion matrix components 
        (TP, TN, FP, FN).  
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              colors : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
			  color_mask : Array  - Color-coded image of the confusion matrix components.
    """
    # Set color scheme
    if color_scheme == 'CBMY':
      cm_color = {'tp':(0, 255, 255), 'tn':(0, 0, 0),'fp':(255, 0, 255),'fn':(255, 255, 0)} # TP: Cyan, TN: Black, FP: Magenta, FN:Yellow
    elif color_scheme == 'GBBR':
      cm_color = {'tp':(0, 255, 0), 'tn':(0, 0, 0),'fp':(0, 0, 255),'fn':(255, 0, 0)} # TP: Green, TN: Black, FP: Blue, FN: Red
    # Convert input tensor images to 2D arrays and apply segmentation threshold
    y_t = y_true[:,:,0]
    y_p = y_pred[:,:,0]>threshold
    # Image shape for the color mask
    shape = (int(y_t.shape[0]),int(y_t.shape[1]),3)
    # Import confusion matrix components 2D boolean masks 
    masks = confusion_matrix_arrays(y_t, y_p)
    # Compute color mask from confusion matrix 2D boolean arrays
    color_mask = np.zeros(shape,dtype=np.uint8)
    for l, mask in masks.items():
      color = cm_color[l]
      mask_rgb = np.zeros(shape,dtype=np.uint8)
      mask_rgb[mask != 0] = color
      color_mask += mask_rgb
    return (color_mask)

def confusion_matrix_arrays(y_true, y_pred):
    """ Returns a dictionnary containing the confusion matrix 
        components (TP, FP, FN, TN) as 2D boolean arrays.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation mask.
              y_pred : Tensor - Predicted probability map.
              threshold : Float - Threshold for binary segmentation of the probability map.
    ----------
    Returns
    ----------
	          cm_arrays : Dictionnary - Confusion matrix components as 2D boolean arrays.
    """
    # Create a dictionnary storing confusion matrix arrays
    cm_arrays = {}
    # Use 'logical not' to create inverses of groundtruth and predicted masks 
    groundtruth_inverse = np.logical_not(y_true)
    predicted_inverse = np.logical_not(y_pred)
    # Use 'logical and' to compute 2D confusion matrix boolean arrays
    cm_arrays['tp'] = np.logical_and(y_true, y_pred)
    cm_arrays['tn'] = np.logical_and(groundtruth_inverse, predicted_inverse)
    cm_arrays['fp'] = np.logical_and(groundtruth_inverse,  y_pred)
    cm_arrays['fn'] = np.logical_and(y_true, predicted_inverse)
    return (cm_arrays)

def confusion_matrix_image_overlay(image, y_true, y_pred, threshold, alpha, color_scheme='CBMY'):
    """ Generate an overlay of the color-coded confusion matrix components (TP,
        TN, FP, FN) on top of the original grayscale image.
    ----------
    Arguments
    ----------
              image : Array - Input image
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              alpha : Float - Transparency parameter (0 : transparent and 1: Opaque)
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
		      overlay_image : Array  - Input image overlayed with color-coded confusion matrix components.
    """
    # Set color scheme
    if color_scheme == 'CBMY':
      cm_color = {'tp':(0, 255, 255), 'tn':(0, 0, 0),'fp':(255, 0, 255),'fn':(255, 255, 0)} # TP: Cyan, TN: Black, FP: Magenta, FN:Yellow
    elif color_scheme == 'GBBR':
      cm_color = {'tp':(0, 255, 0), 'tn':(0, 0, 0),'fp':(0, 0, 255),'fn':(255, 0, 0)} # TP: Green, TN: Black, FP: Blue, FN: Red
    # Convert input tensor image to 2D arrays and apply segmentation threshold
    image = image[:,:,0]
    y_t = y_true[:,:,0]
    y_p = y_pred[:,:,0]>threshold
    # Convert initial grayscale image into a RGB image
    image = img_as_ubyte(image)
    image = cvtColor( image, COLOR_GRAY2RGB)
    # Import confusion matrix components 2D boolean masks 
    masks = confusion_matrix_arrays(y_t, y_p)
    # Compute color-coded mask from confusion matrix components boolean arrays
    color_mask = np.zeros_like(image)
    for l, mask in masks.items():
        color = cm_color[l]
        mask_rgb = np.zeros_like(image)
        mask_rgb[mask != 0] = color
        color_mask += mask_rgb
    overlay_image = addWeighted(image, alpha, color_mask, (1-alpha), 0)
    return (overlay_image)
    
def performance_metrics_bench_df(metrics, metrics_names, bench_names):
    """ Create a Panda dataframe object for further plotting with
        Seaborn or other high level python plotting API.
    ----------
    Arguments
    ----------
              metrics : List of Arrays - Columns: Performance metrics / Lines: Values for each image of the data set
              metrics_names : List - Name of the performance metrics given as strings.
              bench_names : List - Name of image segmentation modalities benchmarked given as strings.
    ----------
    Returns
    ----------
              df_bench_metrics : Pandas DF - Dataframe of the performance metrics for each image of the data set.
    """
    # Create a data frame grouping results from the different seg modalities for that given metrics
    l_m1=[]
    for i in range(0, len(metrics_names)):
      l_f = []
      for j in range(0, len(bench_names)):
        df_a=pd.DataFrame(metrics[j][:,i])
        df_b=df_a.assign(Seg = bench_names[j])
        l_f.append(df_b)
      l_m1.append(pd.concat(l_f))
    # Assign a metrics name to each data frame
    l_m2=[]
    for i in range(0, len(metrics_names)):
      l_m2.append(l_m1[i].assign(Metrics = metrics_names[i]))
    # Concatenate named data frames in a single data frame
    cdf = pd.concat(l_m2)         
    # Create final data frame
    df_bench_metrics = pd.melt(cdf, id_vars=['Metrics','Seg']).drop('variable', axis=1)       
    return (df_bench_metrics)
    
###### Instance segmentation ###### 
    
def object_IoU(y_true_l, y_pred_l):
    """ Compute IoU for all pair of labelled objects in the input ground truth 
        and predicted labelled masks and store them in an array.
    ---------
    Arguments
    ---------
             y_true_l : Array - Ground truth labelled mask 
             y_pred_l : Array - Predicted labelled mask.
    ---------
    Returns
    ---------
             IoU : Array - Object-wise IoU matrix for all objects in input masks.
    """
    # Count objects labelled in ground truth and predicted masks
    true_objects = len(np.unique(y_true_l))
    pred_objects = len(np.unique(y_pred_l))

    # Compute area of objects in ground truth and predicted masks
    area_true = np.histogram(y_true_l, bins=true_objects)[0]
    area_pred = np.histogram(y_pred_l, bins=pred_objects)[0]

    # Compute intersection between objects in ground truth and predicted masks
    h = np.histogram2d(y_true_l.flatten(), y_pred_l.flatten(), bins=(true_objects,pred_objects))
    intersection = h[0]
    
    # Calculate union between objects in ground truth and predicted masks
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)
    union = area_true + area_pred - intersection
    
    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9 # To avoid dividing by 0!
    
    # Compute intersection over union  
    IoU = (intersection/union)

    return (IoU)

def get_object_metrics(threshold, IoU):
    """ Compute object-wise metrics: TP, FP, FN, Precision, Recall and F1 score 
        for matching objects i.e pair of objects in ground truth and prediction 
        masks whose IoU is above a certain threshold.
    ---------
    Arguments
    ---------
             IoU : Array - Object-wise IoU matrix for all objects in input masks
             threshold : Float - IoU threshold for defining matching objects between two masks.
    ---------
    Returns
    ---------
             F1, P, R, TP, FP, FN : Arrays - F1, P, R, TP, FP, FN for all matching objects.
    """
    
    # Matching objects (i.e pairs of objects from GT and prediction masks with objectwise IoU above threshold)
    matches = IoU > threshold
    
    # Compute TP, FP and FN for matching objects
    true_positives = np.sum(matches, axis=1) == 1   # i.e correct objects
    false_positives = np.sum(matches, axis=0) == 0  # i.e extra objects
    false_negatives = np.sum(matches, axis=1) == 0  # i.e missed objects
    
    assert np.all(np.less_equal(true_positives, 1))
    assert np.all(np.less_equal(false_positives, 1))
    assert np.all(np.less_equal(false_negatives, 1))
    
    TP, FP, FN = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)

    # Compute precision for matching objects
    P = TP / (TP + FP + 1e-9) 
    
    # Compute recall for matching objects
    R = TP / (TP + FN + 1e-9 )
    
    # Compute F1 score for matching objects
    F1 = 2*TP / (2*TP + FP + FN + 1e-9)
    
    # Compute fraction of missed objects for matching objects
    FMO = FN / (TP + FN + 1e-9 )
    
    # Compute fraction of missed objects for matching objects
    FEO = FP / (TP + FN + 1e-9 )
    
    return (F1, P, R, TP, FP, FN, FMO, FEO)

def instance_segmentation_performance_metrics(y_true_l, y_pred_l):
    """ Compute object-wise metrics: TP, FP, FN and F1 score for matching
        objects at all IoU threshold.
    ---------
    Arguments
    ---------
             y_true_l : Array - Ground truth labelled mask 
             y_pred_l : Array - Predicted labelled mask.
    ---------
    Returns
    ---------
             F1, P, R, TP, FP, FN, FMO, FEO : Arrays - F1, P, R, TP, FP, FN, FMO, FEO for all matching objects.
    """
    # Create a Pandas data frame to store results
    df_results = pd.DataFrame(columns=["Image", "Threshold", "mIoU", "F1", "Precision", "Recall", "TP", "FP", "FN", "FMO", "FEO"])
    
    # Compute for each image object-wise metrics at different IoU thresholds
    for i in range(0, len(y_true_l)):
      # Compute object-wise IoU object matrix and compute the mean IoU for all objects in an image
      IoU = object_IoU(y_true_l[i], y_pred_l[i])
      if IoU.shape[0] > 0:
        jac = np.max(IoU, axis=0).mean()
      else:
        jac = 0.0
      for t in np.arange(0.05, 0.99, 0.01):
        f1, p, r, tp, fp, fn, fmo, feo = get_object_metrics(t, IoU)
        results = {"Image": i, "Threshold": t, "mIoU": jac, "F1": f1, "Precision": p, "Recall": r, "TP": tp, "FP": fp, "FN": fn, "FMO": fmo, "FEO": feo}
        r_index = df_results.shape[0]+1 
        df_results.loc[r_index] = results
        
    return (df_results)

###### Plots ###### 

def precision_recall_curve_plot(y_true, y_pred, path, file_name, x_range=None, y_range=None, fig_close=False):
    """ Plot precision-recall curves for various thresholds for an image set 
        and display the optimal threshold for probability maps binarization.
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation masks
              y_pred : Tensor - Predicted probability maps
              path : String - Path to save file on drive
              file_name : String - File name
              x_range: List - Range [min, max] of the x axis
              y_range: List - Range [min, max] of the y axis
              fig_close: Boolean - Don't display the figure plot (useful for memory management).
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Compute data 
    precision, recall, thresholds, F1_score, ix, B_T, F1_T = precision_recall_curve_data(y_true, y_pred)
    # Plot
    fig,axes = plt.subplots(1,1,figsize=(5.5,5.5))
    # Plot iso-F1 score lines
    f_scores = np.linspace(0.4, 0.9, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1.05)
        y = f_score * x / (2 * x - f_score) # Precision (y) as function of recall (x) and taking F1 score as a parameter
        l, = axes.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.75)
        axes.annotate('F1={0:0.1f}'.format(f_score), xy=(0.89, y[45] - 0.04), color='gray', alpha=0.9)
    lines.append(l)
    labels.append('F1 score isolines')
    # Plot precision-recall curve 
    l, = axes.plot(precision, recall, linestyle='-', color='black', linewidth=2.0, alpha=0.9)
    lines.append(l)
    labels.append('Precision-recall curve')
    # Plot best threshold
    l = axes.scatter( precision[ix], recall[ix], marker='o', s=50.0, color='red', alpha=0.9)
    lines.append(l)
    labels.append('Best threshold = %0.2f (F1 = %0.2f)' % ( B_T, F1_T))
    #Plot legend
    axes.legend(lines, labels, fontsize=12)
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    if y_range == None:
      axes.set_xlim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_xlim(x_range)
    axes.set_xlabel(r'Recall', fontsize=18)
    axes.set_ylabel(r'Precision', fontsize=18)
    axes.tick_params(axis='both', which='major', labelsize=14)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Precision_Recall_Plot.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Precision_Recall_Plot.png', dpi=300)
    if fig_close == True: 
        plt.close(fig)
    else:
        pass
    return

def confusion_matrix_plot(y_true, y_pred, threshold, classes, path, file_name, im_set=True, norm=True, title=None, fig_close=False, cmap=plt.cm.Blues):
    """ Plot & save the confusion matrix.
    ----------
    Parameters
    ----------
			  y_true : Tensor - Ground truth segmentation masks
              y_pred : Tensor - Predicted probability map              
              threshold : Float - Threshold for binary segmentation of the probability map
              classes : String - Name to be given to both classes ([0,1] or ['BCKD','FRGD'])
              path : String - Path to save file on drive
              file_name : String - File name 
              im_set : Boolean - To choose between image set or single image
              norm : Boolean - To choose between the raw or normalised confusion matrix
              title : String - Title of the plot. None sets default title
              fig_close: Boolean - Don't display the figure plot (useful for memory management)
              cmap : String - Colourmap used for the plot.
    ----------
    Returns
    ----------
			  Fig : Matplotlib object - Plot.
    """
    if im_set == False:
      if not title:
        if norm == True:
          title = 'Normalised confusion matrix'
        else:
          title = 'Raw confusion matrix'
      # Compute confusion matrix
      cm = confusion_matrix_set(y_true, y_pred, threshold, im_set, norm)
      # Plot confusion matrix 
      fig, axes = plt.subplots(figsize=(5.0, 5.0))
      im = axes.imshow(cm, interpolation='nearest', cmap=cmap)
      fig.colorbar(im,ax=axes,cax=make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%"))
      axes.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes)
      axes.set_ylabel('Ground truth', fontsize=14)
      axes.set_xlabel('Prediction', fontsize=14)
      axes.set_title(title, fontsize=14)
      plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # Rotate the tick labels and set their alignment.
      # Loop over data dimensions and create text annotations.
      thresh = np.divide(np.max(cm),2.0)
      for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
          axes.text(j, i, (r'%0.2f' % cm[i, j] ), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
      fig.tight_layout()
      fig.show()
      fig.savefig(path + file_name +'_Confusion_Matrix.pdf', dpi=300) # Save the plot 
      # fig.savefig(path + file_name +'_Confusion_Matrix.png', dpi=300)
      if fig_close ==True:
          plt.close(fig)
      else:
        pass
    else:
      if not title:
          if norm == True:
              title = 'Normalised confusion matrix'
          else:
              title = 'Raw confusion matrix'
      # Compute average confusion matrix and std
      cm, cm_std = confusion_matrix_set(y_true, y_pred, threshold, norm, im_set)
      # Plot confusion matrix 
      fig, axes = plt.subplots(figsize=(5.0, 5.0))
      im = axes.imshow(cm, interpolation='nearest', cmap=cmap)
      fig.colorbar(im,ax=axes,cax=make_axes_locatable(plt.gca()).append_axes("right", "5%", pad="3%"))
      axes.set(xticks=np.arange(cm.shape[1]),yticks=np.arange(cm.shape[0]), xticklabels=classes, yticklabels=classes)
      axes.set_ylabel('Ground truth', fontsize=14)
      axes.set_xlabel('Prediction', fontsize=14)
      axes.set_title(title, fontsize=14)
      plt.setp(axes.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor") # Rotate the tick labels and set their alignment.
      # Loop over data dimensions and create text annotations.
      thresh = np.divide(np.max(cm),2.0)
      for i in range(0, cm.shape[0]):
        for j in range(0, cm.shape[1]):
          axes.text(j, i, (r'%0.2f' % cm[i, j] + r'$\pm %0.2f$' % cm_std[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
      fig.tight_layout()
      fig.show()
      fig.savefig(path + file_name +'_Average_Confusion_Matrix.pdf', dpi=300) # Save the plot 
      # fig.savefig(path + file_name +'_Average_Confusion_Matrix.png', dpi=300)
      if fig_close == True:
          plt.close(fig)
      else:
          pass
    return

def performance_metrics_boxplots(metrics, metrics_names, path, file_name, y_range=None, color_scheme='Set1', fig_close=False):
    """ Plot and save boxplots of the various performance metrics for a given
        model /image set.
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings
              path : String - Path to save file on drive
              file_name : String - File name 
              y_range : List - Set y-axis value range
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Compute performance metrics dataframe
    df_perf_metrics = sem_perf_metrics_plot_df(metrics, metrics_names)
    # Plot overlay image 
    fig,axes = plt.subplots(1,1,figsize=(5.5,5.0))
    axes= sns.swarmplot(x="Metrics", y="value", hue="Metrics", data=df_perf_metrics, size=2.0, color='0.25', alpha=0.75, zorder=0)
    axes=sns.boxplot(x="Metrics", y="value", hue="Metrics", data=df_perf_metrics, color=None, palette=color_scheme, dodge=False, width=0.6, whis=1.5, showmeans=True, meanline=True,
    showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.75), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5,
    edgecolor='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.75), meanprops=dict(linestyle='--', linewidth=1.5, color='red', 
    alpha=0.75),zorder=1)
    handles,_ = axes.get_legend_handles_labels()
    axes.legend(handles, metrics_names, fontsize=14)  
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    axes.set_ylabel(r'Performance metrics values', size=18)
    axes.tick_params(axis='both', which='both', direction='in', bottom=False, labelbottom=False, labelsize=14)
    axes.set(xlabel=None)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Performance_Metrics_Boxplot.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Performance_Metrics_Boxplot.png', dpi=300) # Save the plot
    if fig_close == True: 
        plt.close(fig)
    else:
        pass    
    return

def performance_metrics_barplots(metrics, metrics_names, path, file_name, y_range=None, color_scheme='Set1'):
    """ Plot and save barplots of the various performance metrics for a given
        model/image set.
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings
              path : String - Path to save file on drive
              file_name : String - File name 
              y_range : List - Set y-axis value range
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Compute performance metrics dataframe
    df_perf_metrics = sem_perf_metrics_df(metrics, metrics_names)
    # Plot overlay image 
    fig,axes = plt.subplots(1,1,figsize=(7.0,5.0))
    axes=sns.barplot(x="Metrics", y="value", hue="Metrics", data=df_perf_metrics, dodge=False, color=None, palette=color_scheme, saturation = 0.75,  capsize = 0.075, errcolor = 'gray', errwidth = 2.0, ci = 'sd' )
    handles,_ = axes.get_legend_handles_labels()
    axes.legend(handles, metrics_names, fontsize=14, loc='upper left', bbox_to_anchor=(1, 0.5))
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    axes.set_ylabel(r'Performance metrics values', size=18)
    axes.tick_params(axis='both', which='both', direction='in', bottom=False, labelbottom=False, labelsize=14)
    axes.set(xlabel=None)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Performance_Metrics_Barplot.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Performance_Metrics_Barplot.png', dpi=300) # 
    return 

def confusion_matrix_image_plot(y_true, y_pred, threshold, path, file_name, color_scheme='CBMY'):
    """ Plot a color-coded image of the confusion matrix components(TP, TN, FP, FN).  
    ----------
    Arguments
    ----------
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              path : String - Path to save file on drive
              file_name : String - File name 
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          color_mask : Array  - Color-coded image of the confusion matrix components.
    """
    # Compute overlay image
    cm_im=confusion_matrix_image(y_true, y_pred, threshold)

    # Plot overlay image 
    fig,axes = plt.subplots(figsize=(4.0,4.0))
    axes.imshow(cm_im)
    axes.set_title('Confusion matrix', fontsize=14)
    axes.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Confusion_Matrix_Image_Plot.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Confusion_Matrix_Image_Plot.png', dpi=300)
    return 

def confusion_matrix_image_overlay_plot(image, y_true, y_pred, threshold, alpha, path, file_name, color_scheme='CBMY'):
    """ Plot and save an overlay of the color-coded confusion matrix components 
        (TP, TN, FP, FN) on top of the original grayscale image.
    ----------
    Arguments
    ----------			  
              image : Array - Input image
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              alpha : Float - Transparency parameter (0 : transparent and 1: Opaque)
              path : String - Path to save file on drive
              file_name : String - File name 
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          overlay_image : Array  - Input image overlayed with color-coded confusion matrix components.
    """
    # Compute overlay image
    overlay_im=confusion_matrix_image_overlay(image, y_true, y_pred, threshold, alpha, color_scheme)

    # Plot overlay image 
    fig,axes = plt.subplots(figsize=(4.0,4.0))
    axes.imshow(overlay_im)
    axes.set_title('Confusion matrix image overlay', fontsize=14)
    axes.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Overlay_Confusion_Matrix_Plot.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Overlay_Confusion_Matrix_Plot.png', dpi=300)
    return 

def example_evaluation_semantic_seg_images(image, y_true, y_pred, threshold, alpha, path, file_name, N_im, S_im, color_scheme='CBMY'):
    """ Plot and save sample image / ground thruth mask / predictions map / 
        segmentation mask / confusion matrix / confusion matrix overlayed image. 
    ----------
    Arguments
    ----------			  
              image : Array - Input image
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              threshold : Float - Threshold for binary segmentation of the probability map
              alpha : Float - Transparency parameter (0 : transparent and 1: Opaque)
              path : String - Path to save file on drive
              file_name : String - File name 
              N_im : Float - Number of images to be plotted
              S_im : Integer - Random seed
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    fig,axes = plt.subplots(N_im,6,figsize=(19.5, (N_im * 3.25)))
    for i in range (0,N_im):
      j = S_im + i
      axes[i,0].plot
      axes[i,0].imshow(image[j][:,:,0], cmap = 'gray', origin = 'lower') # Image patch 
      axes[i,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,0].set_title('Image', fontsize=14)
      axes[i,1].plot
      axes[i,1].imshow(y_true[j][:,:,0], cmap = 'gray', origin = 'lower') # Ground truth mask
      axes[i,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,1].set_title('Ground truth', fontsize=14)
      axes[i,2].plot
      axes[i,2].imshow(y_pred[j][:,:,0], cmap = 'magma' , origin = 'lower') # Prediction (probaility in [0;1]) 
      axes[i,2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,2].set_title('Prediction', fontsize=14)
      axes[i,3].plot
      axes[i,3].imshow((y_pred[j][:,:,0]>threshold), cmap = 'gray', origin = 'lower')# Binary segmentation mask
      axes[i,3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,3].set_title('Binary mask', fontsize=14)
      axes[i,4].plot
      axes[i,4].imshow(confusion_matrix_image(y_true[j], y_pred[j], threshold), origin = 'lower')# Confusion matrix image
      axes[i,4].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,4].set_title('Confusion matrix', fontsize=14)
      axes[i,5].plot
      axes[i,5].imshow(confusion_matrix_image_overlay(image[j], y_true[j], y_pred[j], threshold, alpha), origin = 'lower')# Confusion matrix image overlay
      axes[i,5].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,5].set_title('Confusion matrix overlay', fontsize=14)  
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Model_Prediction_Results_Images.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Model_Prediction_Results_Images.png', dpi=300)
    return

def example_benchmarking_masks(images, y_true, y_pred, model_name, threshold, bench_masks, bench_names, path, file_name, s_im):
    """ Plot benchmarking masks for a given example image/ground truth mask.  
    ----------
    Arguments
    ----------
              images : Array - Test images.
              y_true : Array - Image ground truth segmentation masks
              y_pred : Tensor - Predicted probability maps by the benchmarked model
              model_name : String - Name of the benchmarked model
              threshold : Float - Threshold for binary segmentation of the probability map
              bench_masks : Array - Array containing benchmark segmentation masks
              bench_names : List - Name of image segmentation modalities used to obtain benchmark segmentation masks
              path : String - Path to save file on drive
              file_name : String - File name.
              s_im : Integer - Random seed
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Number of image segmentation modalities benchmarked 
    N_bench = len(bench_names)
    # Number of panels to be plotted
    N_panel = 3 + N_bench
    # Generate binary mask out of predicted probability map
    model_mask = y_pred[s_im][:,:,0]>threshold
    # Plot example
    fig,axes = plt.subplots(1,N_panel,figsize=((2.75*N_panel),3.0))
    axes[0].plot
    axes[0].imshow(images[s_im][:,:,0], 'gray', origin = 'lower')
    axes[0].set_title('Image', fontsize=16)
    axes[0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[1].plot
    axes[1].imshow(y_true[s_im][:,:,0], 'gray', origin = 'lower')
    axes[1].set_title('Ground truth mask', fontsize=16)
    axes[1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[2].plot
    axes[2].imshow(model_mask, 'gray', origin = 'lower')
    axes[2].set_title(model_name, fontsize=16)
    axes[2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    for i in range (0,N_bench):
      j= 3 + i 
      axes[j].plot
      axes[j].imshow(bench_masks[i][s_im], 'gray', origin = 'lower')
      axes[j].set_title(bench_names[i], fontsize=16)
      axes[j].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Example_Benchmarking_Masks.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Example_Benchmarking_Masks.png', dpi=300)
    return 

def benchmarking_results_image_plot(image, y_true, y_pred, model_name, threshold, bench_masks, bench_names, alpha, path, file_name, s_im, color_scheme='CBMY'):
    """ Plot and save sample image / ground thruth mask / segmentation mask for 
        a given benchmarked modality and associated confusion matrix / confusion 
        matrix overlayed image. 
    ----------
    Arguments
    ----------			  
              image : Array - Input image
              y_true : Tensor - Ground truth segmentation mask 
              y_pred : Tensor - Predicted probability map
              model_name : String - Name of the benchmarked model
              threshold : Float - Threshold for binary segmentation of the probability map
              bench_masks : Array - Array containing benchmark segmentation masks
              bench_names : List - Name of image segmentation modalities used to obtain benchmark segmentation masks
              alpha : Float - Transparency parameter (0 : transparent and 1: Opaque)
              path : String - Path to save file on drive
              file_name : String - File name 
              N_im : Float - Number of images to be plotted
              S_im : Integer - Random seed
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Number of lines in the plot array
    N_lines = len(bench_names) + 1
    # Plot
    fig,axes = plt.subplots(N_lines,5,figsize=(16.0,(N_lines * 3.3)))
    axes[0,0].plot
    axes[0,0].imshow(image[s_im][:,:,0], cmap = 'gray', origin = 'lower') # Image patch 
    axes[0,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0,0].set_title('Image', fontsize=14)
    axes[0,1].plot
    axes[0,1].imshow(y_true[s_im][:,:,0], cmap = 'gray', origin = 'lower') # Ground truth mask
    axes[0,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0,1].set_title('Ground truth', fontsize=14)
    axes[0,2].plot
    axes[0,2].imshow((y_pred[s_im][:,:,0]>threshold), cmap = 'gray', origin = 'lower')# Binary segmentation mask
    axes[0,2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0,2].set_title(model_name, fontsize=14)
    axes[0,3].plot
    axes[0,3].imshow(confusion_matrix_image(y_true[s_im], y_pred[s_im], threshold), origin = 'lower')# Confusion matrix image
    axes[0,3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0,3].set_title('Confusion matrix', fontsize=14)
    axes[0,4].plot
    axes[0,4].imshow(confusion_matrix_image_overlay(image[s_im], y_true[s_im], y_pred[s_im], threshold, alpha), origin = 'lower')# Confusion matrix image overlay
    axes[0,4].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    axes[0,4].set_title('Confusion matrix overlay', fontsize=14)  
    for i in range (1,N_lines):
      j = i-1
      axes[i,0].plot
      axes[i,0].imshow(image[s_im][:,:,0], cmap = 'gray', origin = 'lower') # Image patch 
      axes[i,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,0].set_title('Image', fontsize=14)
      axes[i,1].plot
      axes[i,1].imshow(y_true[s_im][:,:,0], cmap = 'gray', origin = 'lower') # Ground truth mask
      axes[i,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,1].set_title('Ground truth', fontsize=14)
      axes[i,2].plot
      axes[i,2].imshow(bench_masks[j][s_im][:,:,0], cmap = 'gray', origin = 'lower')# Binary segmentation mask
      axes[i,2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,2].set_title(bench_names[j], fontsize=14)
      axes[i,3].plot
      axes[i,3].imshow(confusion_matrix_image(y_true[s_im],bench_masks[j][s_im], threshold), origin = 'lower')# Confusion matrix image
      axes[i,3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,3].set_title('Confusion matrix', fontsize=14)
      axes[i,4].plot
      axes[i,4].imshow(confusion_matrix_image_overlay(image[s_im], y_true[s_im], bench_masks[j][s_im], threshold, alpha), origin = 'lower')# Confusion matrix image overlay
      axes[i,4].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,4].set_title('Confusion matrix overlay', fontsize=14)  
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Model_Benchmarking_Results_Images.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Model_Benchmarking_Results_Images.png', dpi=300)
    return

def benchmark_metrics_boxplots(metrics, metrics_names, bench_names, path, file_name, y_range=None, color_scheme='Set1'):
    """ Plot and save boxplots of the various performance metrics for the
        whole benchmarking set.    
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings
              bench_names : List - Name of image segmentation modalities benchmarked given as strings
              path : String - Path to save file on drive
              file_name : String - File name 
              y_range : List - Set y-axis value range
              color_scheme : Dictionnary - Color scheme used for display.
    ----------
    Returns
    ----------
              Fig : Matplotlib object - Plot.
    """
    # Compute benchmarking performance metrics dataframe
    df_bench_metrics = performance_metrics_bench_df(metrics, metrics_names, bench_names)

    # Plot
    fig,axes = plt.subplots(1,1,figsize=(15.0,6.0))
    axes=sns.swarmplot(x="Metrics", y="value", hue="Seg", data=df_bench_metrics, dodge=True, size=1.0, color='0.25', alpha=0.75, zorder=0)
    axes=sns.boxplot(x="Metrics", y="value", hue="Seg", data=df_bench_metrics, palette=color_scheme, dodge=True,  whis=1.5, showmeans=True, meanline=True,
    showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.75), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5,
    edgecolor='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.75), meanprops=dict(linestyle='--', linewidth=1.5, color='red', 
    alpha=0.75),zorder=1)
    handles,_ = axes.get_legend_handles_labels()
    axes.legend(handles, bench_names, fontsize=14)  
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    axes.set_ylabel(r'Performance metrics values', size=18)
    axes.set(xlabel=None)
    axes.set_xticklabels(metrics_names)
    axes.tick_params(axis='both', which='both', direction='in', bottom=True, labelbottom=True, labelsize=14)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Benchmarking_Boxplots.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Benchmarking_Boxplots.png', dpi=300) # Save the plot 
    return

def indiv_benchmark_metrics_boxplots(metrics, metrics_names, bench_names, select_metrics, path, file_name, y_range=None, color_scheme='Set2'):
    """ Plot and save boxplots of a given performance metrics on the whole 
        benchmarking data set.
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings
              bench_names : List - Name of image segmentation modalities benchmarked given as strings
              select_metrics : String - Name of the performance metrics selected to be plotted 
              path : String - Path to save file on drive
              file_name : String - File name 
              y_range : List - Set y-axis value range
              color_scheme : Dictionnary - Color scheme used for display.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Compute benchmarking performance metrics dataframe
    df_bench_metrics = performance_metrics_bench_df(metrics, metrics_names, bench_names)

    # Select a particular performance metrics
    condition = (df_bench_metrics['Metrics'] == select_metrics) 
    df_indiv = df_bench_metrics[condition]

    # Plot
    fig,axes = plt.subplots(1,1,figsize=(5.5,5.0))
    axes= sns.swarmplot(x="Seg", y="value", hue="Seg", data= df_indiv, dodge=False, size=2.0, color='0.25', alpha=0.75, zorder=0)
    axes=sns.boxplot(x="Seg", y="value", hue="Seg", data= df_indiv, color=None, palette=color_scheme, dodge=False, width=0.6, whis=1.5, showmeans=True, meanline=True,
    showfliers=False, whiskerprops=dict(linewidth=1.5, color='black', alpha=0.75), capprops=dict(linewidth=1.5, color='black', alpha=0.7), boxprops=dict(linewidth=1.5,
    edgecolor='black', alpha=0.75), medianprops=dict(linestyle='-', linewidth=1.5, color='black',alpha=0.75), meanprops=dict(linestyle='--', linewidth=1.5, color='red', 
    alpha=0.75),zorder=1)
    handles,_ = axes.get_legend_handles_labels()
    axes.legend(handles, bench_names, fontsize=14)  
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    axes.set_ylabel(select_metrics, size=18)
    axes.tick_params(axis='both', which='both', direction='in', bottom=False, labelbottom=False, labelsize=14)
    axes.set(xlabel=None)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name + select_metrics +'_Benchmarking_Boxplots.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name + select_metrics +'_Benchmarking_Boxplots.png', dpi=300) # Save the plot 
    return

def indiv_benchmark_metrics_barplots(metrics, metrics_names, bench_names, select_metrics, path, file_name, y_range=None, color_scheme='Set2'):
    """ Plot and save barplots of a given performance metrics on the whole 
        benchmarking data set.
    ----------
    Arguments
    ----------
              metrics : Array - Columns: Performance metrics & Lines: Values for each image of the data set
              metrics_name : List - Name of the performance metrics given as strings
              bench_names : List - Name of image segmentation modalities benchmarked given as strings
              select_metrics : String - Name of the performance metrics selected to be plotted 
              path : String - Path to save file on drive
              file_name : String - File name 
              y_range : List - Set y-axis value range
              color_scheme : Dictionnary - Color scheme used to display confusion matrix components.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Compute benchmarking performance metrics dataframe
    df_bench_metrics = performance_metrics_bench_df(metrics, metrics_names, bench_names)

    # Select a particular performance metrics
    condition = (df_bench_metrics['Metrics'] == select_metrics) 
    df_indiv = df_bench_metrics[condition]

    # Plot
    fig,axes = plt.subplots(1,1,figsize=(7.0,5.0))
    axes=sns.barplot(x="Seg", y="value", hue="Seg", data=df_indiv, dodge=False, color=None, palette=color_scheme, saturation = 0.75,  capsize = 0.075, errcolor = 'gray', errwidth = 2.0, ci = 'sd' )
    handles,_ = axes.get_legend_handles_labels()
    axes.legend(handles, bench_names, fontsize=14, loc='upper left', bbox_to_anchor=(1, 0.36))
    if y_range == None:
      axes.set_ylim(auto=True, ymin=None, ymax=None) 
    else:
      axes.set_ylim(y_range)
    axes.set_ylabel(select_metrics, size=18)
    axes.tick_params(axis='both', which='both', direction='in', bottom=False, labelbottom=False, labelsize=14)
    axes.set(xlabel=None)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name + select_metrics +'_Benchmarking_Barplots.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name + select_metrics +'_Benchmarking_Barplots.png', dpi=300) # Save the plot 
    return

def example_evaluation_instance_seg_images(image, y_true_b, y_pred_b, y_true_l, y_pred_l, path, file_name, N_im, S_im):
    """ Plot and save sample image / ground thruth binary mask / predicted 
        binary mask / ground truth labelled mask / predicted labelled mask. 
    ----------
    Arguments
    ----------			  
              image : Tensor - Input image
              y_true_b : Tensor - Ground truth binary mask 
              y_pred_b : Tensor - Predicted binary mask
              y_true_l : Array - Ground truth labelled mask 
              y_pred_l : Array - Predicted labelled mask
              path : String - Path to save file on drive
              file_name : String - File name 
              N_im : Float - Number of images to be plotted
              S_im : Integer - Random seed.
    ----------
    Returns
    ----------
	          Fig : Matplotlib object - Plot.
    """
    # Shuffled spectral colormap
    vals = np.linspace(0,1,256)
    np.random.shuffle(vals)
    cmap_shuf = plt.cm.colors.ListedColormap(plt.cm.nipy_spectral(vals))
    # Object colormap (similar to Fiji glasbey)
    spec = cc.cm.glasbey
    newcolors = spec(np.linspace(0, 1, 256))
    black = np.array([0, 0, 0, 1])
    newcolors[:2, :] = black
    glas_obj = plt.cm.colors.ListedColormap(newcolors)
    #Plot
    fig,axes = plt.subplots(N_im,5,figsize=(16.25, (N_im * 3.25)))
    for i in range (0,N_im):
      j = S_im + i
      axes[i,0].plot
      axes[i,0].imshow(image[j][:,:,0], cmap = 'gray', origin = 'lower') # Image patch 
      axes[i,0].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,0].set_title('Image', fontsize=14)
      axes[i,1].plot
      axes[i,1].imshow(y_true_b[j][:,:,0], cmap = 'gray', origin = 'lower') # Ground truth binary mask
      axes[i,1].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,1].set_title('Ground truth binary mask', fontsize=14)
      axes[i,2].plot
      axes[i,2].imshow(y_pred_b[j][:,:,0], cmap = 'gray' , origin = 'lower') # Predicted binary mask
      axes[i,2].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,2].set_title('Predicted binary mask', fontsize=14)
      axes[i,3].plot
      axes[i,3].imshow(y_true_l[j], cmap = glas_obj, origin = 'lower', interpolation='Nearest')  # Ground truth labelled mask
      axes[i,3].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,3].set_title('Ground truth labelled mask', fontsize=14)
      axes[i,4].plot
      axes[i,4].imshow(y_pred_l[j], cmap = glas_obj, origin = 'lower', interpolation='Nearest') # Predicted labelled mask
      axes[i,4].tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
      axes[i,4].set_title('Predicted labelled mask', fontsize=14)
    fig.tight_layout()
    fig.show()
    fig.savefig(path + file_name +'_Model_Label_Results_Images.pdf', dpi=300) # Save the plot 
    # fig.savefig(path + file_name +'_Model_Label_Results_Images.png', dpi=300)
    return

def F_IoU_threshold_plot(F_mean_l, F_std_l, F_name, model_names, path, file_name, x_range=None, y_range=None, color_scheme='Set1'):
    """Plot F quantities (F1, FMO, FEO) vs IoU curves for a [0.05, 0.99] range
       of IoU thresholds for an arbitrary number of models or modalities.
    ----------
    Arguments
    ----------
    F_mean_l : List - Arrays of mean F scores
    F_std_l : List - Arrays of std F scores 
    F_name : String - Name of quantity (F1, FMO, FEO) 
    model_names : List - Names of models as strings
    path : String - Path to save file on drive
    file_name : String - File name
    x_range: List - Range [min, max] of the x axis
    y_range: List - Range [min, max] of the y axis.
    ----------
    Returns
    ----------
    Fig : Matplotlib object - Plot.
    """ 
    with sns.color_palette(color_scheme):
      fig,axes = plt.subplots(1,1,figsize=(5.0,5.0))
      x=np.arange(0.05, 0.99, 0.01)
      for i in range(0, len(F_mean_l)): 
        axes.plot(x, F_mean_l[i], lw=2, alpha=0.8)
        axes.fill_between(x, F_mean_l[i]-F_std_l[i], F_mean_l[i]+F_std_l[i], alpha=0.2)
      axes.plot([0.7,0.7], [-0.5, 1.5] , lw=2, linestyle='--', color='grey', alpha=0.8)
      axes.legend(model_names, fontsize=14) 
      axes.set_ylim(y_range)
      axes.set_xlim(x_range)
      axes.set_ylabel(F_name, size=18)
      axes.set_xlabel('IoU threshold', size=18)
      axes.tick_params(axis='both', which='both', direction='in', labelsize=14)
      fig.savefig(path + file_name + F_name +'_IoU_threshold_Plot.pdf', dpi=300) # Save the plot 
      # fig.savefig(path + file_name +'_F1_IoU_threshold_Plot.png', dpi=300)
    return

##################################################### Prediction ###################################################

###### Image pre-processing functions ######

def remove_obj_boundary(img):
    """Remove objects touching the boundary of the image.
    ----------
    Arguments
    ----------
			  image : Array - input image.
    ---------
    Returns
    ---------
			 new_image : Array - output image.
    """
    img = img.astype(np.int8)
    orImage = sitk.GetImageFromArray(img)
    image = np.copy(img)
    image[1:-1,1:-1] = 0
    sitkImage = sitk.GetImageFromArray(image)
    recons_filter = sitk.BinaryReconstructionByDilationImageFilter()
    sitkImage = recons_filter.Execute(sitkImage,orImage)
    image = sitk.GetArrayFromImage(sitkImage)
    new_image = img - image
    return (new_image)

def mirror_border(image, size): 
    """Re-size image by adding mirror padding at the boundary.
    ----------
    Arguments
    ----------
			  image : Array - input image.
			  size : integer - size in pixels of the region to be mirrored.
    ---------
    Returns
    ---------
			 new_image : Array - output image.
    """
    im_size_h = image.shape[0] # height 
    im_size_w = image.shape[1] #width
    new_image = np.zeros((3*im_size_h,3*im_size_w))
    img = Image.fromarray(image.astype(float))
    mirror_img = PIL.ImageOps.mirror(img)
    mirror_img = np.array(mirror_img)
    new_image[im_size_h:2*im_size_h,:im_size_w] = mirror_img 
    new_image[im_size_h:2*im_size_h,im_size_w:2*im_size_w] = np.array(img)
    new_image[im_size_h:2*im_size_h,2*im_size_w:] = mirror_img 
    img = img.rotate(180)
    new_image[:im_size_h:,:im_size_w] = np.array(img)
    new_image[:im_size_h,2*im_size_w:] = np.array(img) 
    new_image[2*im_size_h:,:im_size_w] = np.array(img) 
    new_image[2*im_size_h:,2*im_size_w:] = np.array(img)
    mirror_img = PIL.ImageOps.mirror(img)
    mirror_img = np.array(mirror_img)
    new_image[:im_size_h,im_size_w:2*im_size_w] =mirror_img 
    new_image[2*im_size_h:,im_size_w:2*im_size_w] =mirror_img 
    index_h = int(np.round((new_image.shape[0]-size)/2))
    index_w = int(np.round((new_image.shape[1]-size)/2))
    new_image = new_image[index_h:index_h+size,index_w:index_w+size]
    return (new_image)
    
def trim_image(image,patch_height,patch_width):
    """Trim image to fit an interger number of patches of a given size 
       in both width and 
    ----------
    Arguments
    ----------
			  image : Array - input image.
			  patch_height : integer - height in pixels of the patches to be generated.
			  patch_width : integer - height in pixels of the patches to be generated.
    ---------
    Returns
    ---------
			 trim_image: Array - output image.
    """
    trim_h = (image.shape[0]-(image.shape[0]//patch_height*patch_height))/2
    if trim_h % 1 == 0:
      trim_h=int(trim_h)
      trim_h_bis = trim_h
    else:
      trim_h=int(trim_h)
      trim_h_bis = int(trim_h) + 1
    trim_w = (image.shape[1]-(image.shape[1]//patch_width*patch_width))/2
    if trim_w % 1 == 0:
      trim_w=int(trim_w)
      trim_w_bis = trim_w
    else:
      trim_w=int(trim_w)
      trim_w_bis = int(trim_w) + 1
    h_s = 0+trim_h
    h_e = image.shape[0]-trim_h_bis
    w_s = 0+trim_w
    w_e = image.shape[1]-trim_w_bis
    trim_image = image[h_s:h_e, w_s:w_e]
    return (trim_image)
    
def generate_image_patches_periodic_bc(im, patch_size, overlap):
    """Generate image patches from a large image when periodic boundary
       conditions (mirroring) can be used at the boundaries.
    ----------
    Arguments
    ----------
			  image : Array - input image.
			  patch_size : integer - size in pixels of the patches to be generated.
			  overlap : integer - size in pixels of the overlap between patches.
    ---------
    Returns
    ---------
			 data_patches : Array - Array containing output patches.
    """
    data_patches = np.zeros((1,patch_size,patch_size,1))
    imagesize_h = int(im.shape[0]) # height of the image
    imagesize_w = int(im.shape[1]) # width of the image
    # Deals with situation where the input image has one of  
    # its dimensions smaller than the choosen patch size.
    if imagesize_h < patch_size and imagesize_w < patch_size:
        aux = mirror_border(im, patch_size)
    elif imagesize_h < patch_size and imagesize_w >= patch_size:
        aux = mirror_border(im, imagesize_w)
        index_h = int(np.round((aux.shape[0]-patch_size)/2))
        aux = aux[index_h:index_h+patch_size]
    elif imagesize_h >= patch_size and imagesize_w < patch_size:
        aux = mirror_border(im, imagesize_h)
        index_w = int(np.round((aux.shape[1]-patch_size)/2))
        aux = aux[:imagesize_h,index_w:index_w+patch_size]
    else:
        aux = im
    # Generate patches for a given overlap 
    imagesize_h = int(aux.shape[0]) # height of the image
    imagesize_w = int(aux.shape[1]) # width of the image    
    dist = int(patch_size - overlap)
    patch_n_h = int(np.floor((imagesize_h - patch_size)/dist))
    patch_n_w = int(np.floor((imagesize_w - patch_size)/dist))
    for nr in range((patch_n_h+1)):
        for nc in range((patch_n_w+1)):
            patch = aux[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            data_patches = np.concatenate((data_patches,patch))
    if (imagesize_w - patch_size) % dist != 0:
        for i in range((patch_n_h+1)):
            # Rows
            patch = aux[i*dist:i*dist + patch_size,-patch_size:]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            data_patches = np.concatenate((data_patches,patch))
    if (imagesize_h - patch_size) % dist != 0:
        for i in range((patch_n_w+1)):
            # Column
            patch = aux[-patch_size:,i*dist:i*dist + patch_size]
            patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
            data_patches = np.concatenate((data_patches,patch))
    if ((imagesize_h - patch_size) % dist != 0) and ((imagesize_w - patch_size) % dist != 0): 
        # Diagonal
        patch = aux[-patch_size:,-patch_size:]
        patch = patch.reshape((1,patch.shape[0],patch.shape[1],1))
        data_patches = np.concatenate((data_patches,patch))                                
    data_patches = data_patches[1:]
    data_patches = data_patches.astype(np.float32)      
    return (data_patches)
    
def reconstruct_image_from_patches_periodic_bc(patches_im, imagesize_h, imagesize_w, patch_size, overlap):
    """Reconstruct a large image from image patcheswhen periodic boundary conditions (mirroring)
       has been used at the boundaries.
    ----------
    Arguments
    ----------
			  patches_im : Array - input patches.
			  imagesize_h : integer - height in pixels of the final image.
			  imagesize_w : integer - width in pixels of the final image.
			  patch_size : integer - size in pixels of the patches to be generated.
			  overlap : integer - size in pixels of the overlap between patches.
    ---------
    Returns
    ---------
			 recons_im : Array - Array containing the reconstructed large image.
    """
    aux = None # Initialize the variable
    recons_im = np.zeros((imagesize_h, imagesize_w))
    if imagesize_h < patch_size and imagesize_w < patch_size:
        index_h = int((patch_size-imagesize_h)/2)
        index_w = int((patch_size-imagesize_w)/2)
        recons_im = patches_im[0,index_h:-index_h, index_w:-index_w,0]      
    elif imagesize_h < patch_size and imagesize_w >= patch_size:
        index_h = int((patch_size-imagesize_h)/2)
        aux = patches_im[:,index_h:-index_h]
    elif imagesize_h >= patch_size and imagesize_w < patch_size:
        index_w = int((patch_size-imagesize_w)/2)
        aux = patches_im[:,:,index_w:-index_w]
    else:
        aux = patches_im
    if aux is not None:
        # Distance between patches
        dist = int(patch_size - overlap)
        # Count the number of patches to tile in each direction
        patch_n_h = int(np.floor((imagesize_h - patch_size)/dist))
        patch_n_w = int(np.floor((imagesize_w - patch_size)/dist))
        # Template will be used to weight the overlaping regions
        #(value =  1 for non overlapping areas and value = n where n areas overlap)
        template = np.zeros((imagesize_h, imagesize_w)) 
        p = 0;
        for nr in range((patch_n_h+1)):
            for nc in range((patch_n_w+1)):
                template[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] = template[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size]+1
                recons_im[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] = recons_im[nr*dist:nr*dist + patch_size,nc*dist:nc*dist + patch_size] + aux[p,:,:,0]
                p = p+1

        if (imagesize_w - patch_size) % dist != 0:
            for i in range((patch_n_h+1)):
                # Rows
                template[i*dist:i*dist + patch_size,-patch_size:] = template[i*dist:i*dist + patch_size,-patch_size:] + 1
                recons_im[i*dist:i*dist + patch_size,-patch_size:] = recons_im[i*dist:i*dist + patch_size,-patch_size:] + aux[p,:,:,0]
                p = p+1
        if (imagesize_h - patch_size) % dist != 0:
            for i in range((patch_n_w+1)):
                # Column
                template[-patch_size:,i*dist:i*dist + patch_size] = template[-patch_size:,i*dist:i*dist + patch_size] + 1
                recons_im[-patch_size:,i*dist:i*dist + patch_size] = recons_im[-patch_size:,i*dist:i*dist + patch_size] + aux[p,:,:,0]
                p = p+1
        if ((imagesize_h - patch_size) % dist != 0) and ((imagesize_w - patch_size) % dist != 0):
            # Diagonal
            template[-patch_size:,-patch_size:] = template[-patch_size:,-patch_size:] + 1
            recons_im[-patch_size:,-patch_size:] = recons_im[-patch_size:,-patch_size:] + aux[p,:,:,0]
            p = p+1
        template = 1/template
        recons_im = np.multiply(recons_im,template)
    return (recons_im)

def mask_set_to_tensor_set(masks, verbose_mode=True):
    """ Normalises image intensity on [0,1] and transform them into 
        tensors of shape (width x height x 1).
    ----------
    Arguments
    ----------
			  mask_patches : Array - list of output binary mask patches
    ---------
    Returns
    ---------
			 Y_t : Tensor - list of output binary mask patches
    """
    # Compute input shape of tensors 
    mask_shape = masks[0].shape
    width = mask_shape[0]
    height = mask_shape[1]
    tensor_shape = ( width, height, 1 )
    # Mask patches 
    Y_t = [x/np.max(x) for x in masks]
    Y_t = [np.reshape( x, tensor_shape ) for x in Y_t]
    Y_t = np.asarray(Y_t, dtype=int)
    if verbose_mode == True :
        print('There are {} tensors available.'.format(len(Y_t)))
    else:
        pass
    return (Y_t)
    
def image_set_to_tensor_set(images, backbone_prep=None):
    """ Normalises image intensity on [0,1] and transform
        it into tensors for inference.
    ----------
    Arguments
    ----------
              images : Array - List of input images
              backbone_prep : String - Name of the encoder backbone.
    ---------
    Returns
    ---------
              X_i : Tensor - List of convert tensors.
    """
    # Compute input shape of tensors 
    image_shape = images[0].shape
    width = image_shape[0]
    height = image_shape[1]
    tensor_shape = ( width, height, 1 )
    if backbone_prep is None:
        X_i = [x/np.max(x) for x in images] # Normalize intensity between 0 and 1
        X_i = [np.reshape(x, tensor_shape ) for x in X_i] # Reshape images into tensor of input shape
        X_i = np.asarray(X_i, dtype=float) # Give the right numerical format to tensors 
    else:
        preprocess_input = get_preprocessing(backbone_prep) # Define image preprocessing function
        X_i = [preprocess_input(x) for x in images] # Normalize intensity between 0 and 1
        X_i = [np.reshape(x, tensor_shape ) for x in X_i] # Reshape images into tensor of input shape
        X_i = np.asarray(X_i, dtype=np.float32) # Give the right numerical format to tensors 
    return (X_i)
    

###### Inference ######

def import_trained_model_pred(model, path, l_rate=None, loss_func=None, verbose_mode=False):
    """ Load trained DL models for prediction on unseen images or fine tuning. 
    ----------
    Arguments
    ----------
              model : String - Name of model to be loaded
              path : String - Path to where the files are stored on drive
              l_rate : Float - Learning rate if compilation is needed
              loss_func : String - Loss function if compilation is needed.
    ----------
    Returns
    ----------
              trained_model: Tensorflow object - Trained tensorflow model ready for inference .
    """
    if l_rate is None and loss_func is None: 
        trained_model = load_model(path + model +'.h5',compile=False)
    else:
        model = load_model(path + model +'.h5',compile=False)
        model.compile(optimizer = Adam(learning_rate = l_rate), loss = loss_func)
        trained_model = model
    if verbose_mode == True :
        print('Trained model is loaded and ready to be used.')
    else:
        pass
    return (trained_model)

def probability_maps_pred(t_model, im_tensors, verbose_mode=False):
      """ Predict segmentation probability maps for a trained model on all 
          the images of a given image set.
      ----------
      Arguments
      ----------
                t_model : Tensorflow ofbject - Compiled trained tensorflow model
                im_tensors : Tensors - Tensor transformed images. 
      ----------
      Returns
      ----------
                p_maps : Array - Probability maps of the input image set for each input trained model. 
      """
      p_maps = t_model.predict(im_tensors)
      if verbose_mode == True :
          print('Probability maps have been computed for {} images.'.format(len(p_maps)))
      else:
          pass
      return (p_maps)

###### Image post-processing functions ###### 

def image_to_binary_mask(image,threshold):
    """ Transform a grayscale image into a binary image using an user defined
        binarisation threshold.
    ----------
    Arguments
    ----------
              image : Array - Input image
              threshold : Float - Threshold for binary segmentation on [0,1].
    ---------
    Returns
    ---------
             mask : Array - Binary mask of the input image.
    """
    im = np.asarray(image)
    im = (im/np.max(im)) # Normalise image on [0,1]
    mask = (im >threshold).astype(int) # Compute binary mask
    return (mask)

def binary_mask_postprocessing(input_mask, sq_size=2, min_size=None):
    """ Binary mask prostprocessing function based on mathematical morphology 
        removing small white objects using an user defined minimum size.
    ----------
    Arguments
    ----------
              input_mask : Array - Input mask. Default is a "raw" binary mask
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel
              min_size : Integer - Minimum area in pixel below which the element is removed. Default is None.
    ---------
    Returns
    ---------
             mask : Array - Postprocessed binary mask of the input.
    """
    im = np.asarray(input_mask)
    # Create structuring element
    struc_elem = square(sq_size) 
    # Apply morphological opening
    mask = binary_opening(im, footprint=struc_elem)
    if min_size is not None:
        mask = remove_small_objects(mask, min_size=min_size)
    else:
      pass
    return (mask)

def label_mask(mask, threshold, sq_size=2, min_size=10, min_pixel=15, im_watershed=False, exclude_border=False):
    """ Transform a binary image in an (object) labelled image using connected component labelling
        and a distance transform based watershed algorithm. 
    ----------
    Arguments
    ----------
              mask : Array - Input mask.
              threshold : Float - Threshold for binary segmentation on [0,1].
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel.
              min_size : Integer - Minimum area in pixel below which a binary element is removed. Default is 10 pixels.
              min_pixel : Integer - Minimum area allowed for an object. Default is 15 pixels.
              im_watershed : Boolean - Apply distance transform seeded watershed transform to separate touching objects. Default is False.
              exclude_border : Boolean - Remove objects touching the boundaries of the image. Default is false.
    ---------
    Returns
    ---------
             label_mask : Array - Label (instance) mask of the input binary mask.
    """
    # Binarise mask 
    b_mask = image_to_binary_mask(mask, threshold)
    # Clean up binary mask 
    cb_mask = binary_mask_postprocessing(b_mask, sq_size, min_size)
    # Detect and label white objects in the image.
    label_image = label(cb_mask, connectivity=2)
    # Apply distance transform based watershed algorithm to separate touching objects
    if im_watershed == True: 
        # Compute distance transform of the image
        distance = ndimage.distance_transform_edt(cb_mask) 
        # Minimum number of pixels separating peaks in a region of `2 * min_distance + 1` (i.e. peaks are separated by at least `min_distance`)
        min_distance = int(np.ceil(np.sqrt(min_pixel / np.pi)))
        # Compute local maxima in the distance transform of the image
        local_max = peak_local_max(distance, indices=False, exclude_border=False,  min_distance=min_distance, labels=label_image)
        # Compute markers ('seeds')
        markers = label(local_max)
        # Apply watershed transform
        label_image = watershed(-distance, markers, mask=cb_mask)
    else:
        pass
    # Remove objects connected to image border
    if exclude_border == True:
        label_image = clear_border(label_image)
    else:
        pass
    # Remove objects such as their area < min_pixel   
    unique, counts = np.unique(label_image, return_counts=True)
    label_image[np.isin(label_image, unique[counts<min_pixel])] = 0
    # Re-label the final object mask
    label_mask, _ , _ = relabel_sequential(label_image, offset=1)
    return (label_mask)

def binary_mask_pred(im_path, im_folder, file_ext, re_size=False, new_shape=None, backbone_prep=None, model_name=None, model_path=None, threshold=0.5, sq_size=2, min_size=10, save_data=False, out_path=None, out_folder=None):
    """ Compute and save binary masks for a particular trained model, threshold, and image data set.
    ----------
    Arguments
    ----------
              im_path : String - Path to the images to be segmented 
              im_folder : String - Name of the folder where the images to be segmented are to be found 
              file_ext : String - File extension (e.g. '.tif')
              re_size : Boolean - Resize or not loaded patches 
              new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches
              model_name : String - Name of the model to be loaded
              model_path : String - Path to the location where the trained model is saved
              threshold : Float - Threshold for binary segmentation on [0,1]
              save_masks : Boolean - To save on disk computed binary masks
              output_path : String - Path to the location where the computed binary masks are saved
              output_folder : String - Name of the ouput folder where the computed binary masks are saved
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel
              min_size : Integer - Minimum area in pixel below which a binary element is removed. Default is 10 pixels
              backbone_prep : String - Name of the encoder backbone.
    ---------
    Returns
    ---------
             binary_masks : Array - Binary masks for the input images / trained model / threshold. 
    """
    # Load images from disk
    images = load_images(im_path, im_folder, file_ext, re_size, new_shape, verbose_mode=False)
    print('Number of imported images: {}.'.format(len(images)))
    # Convert images into tensors for inference
    tensors = image_set_to_tensor_set(images, backbone_prep)
    # Import trained model
    model = import_trained_model_pred(model_name, model_path, l_rate=None, loss_func=None) # No need to have l_rate and loss_func for inference
    # Predict probability maps for the image set
    p_maps = probability_maps_pred(model, tensors)
    # Transform probability maps into binary masks
    binary_masks = []
    for i in range(0,len(tensors)):
        b_mask = image_to_binary_mask(p_maps[i].squeeze(), threshold) # Compute binary mask  
        cb_mask = binary_mask_postprocessing(b_mask, sq_size, min_size)  # Clean up binary mask 
        binary_masks.append(cb_mask)
    print('Number of computed binary masks: {}.'.format(len(binary_masks)))
    # Save computed binary masks
    if save_data == True: 
        filenames = [i for i in os.listdir( im_path + im_folder) if i.endswith(file_ext)] # Load input image file names
        filenames.sort() # Sort input image file names
        for i in range(0,len(binary_masks)):
            io.imsave(out_path + out_folder + 'Binary_Mask_'+ str(filenames[i]), binary_masks[i], plugin='tifffile', check_contrast=False)
        print('Number of binary masks saved: {}.'.format(len(binary_masks)))
    else:
        pass
    return (images, binary_masks)
    
def label_mask_pred(im_path, im_folder, file_ext, re_size=False, new_shape=None, backbone_prep=None, model_name=None, model_path=None, threshold=0.5, sq_size=2, min_size=10, min_pixel=15, im_watershed=False, exclude_border=False, save_data=False, out_path=None, out_folder=None):
    """ Compute and save instance masks for a particular trained model, threshold, and image data set.
    ----------
    Arguments
    ----------
              im_path : String - Path to the images to be segmented 
              im_folder : String - Name of the folder where the images to be segmented are to be found 
              file_ext : String - File extension (e.g. '.tif')
              re_size : Boolean - Resize or not loaded patches 
              new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches
              model_name : String - Name of the model to be loaded
              model_path : String - Path to the location where the trained model is saved
              threshold : Float - Threshold for binary segmentation on [0,1]
              backbone_prep : String - Name of the encoder backbone.
              save_masks : Boolean - To save on disk computed binary masks
              output_path : String - Path to the location where the computed binary masks are saved
              output_folder : String - Name of the ouput folder where the computed binary masks are saved
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel
              min_size : Integer - Minimum area in pixel below which a binary element is removed. Default is 10 pixels
              min_pixel : Integer - Minimum area allowed for an object. Default is 15 pixels.
              im_watershed : Boolean - Apply distance transform seeded watershed transform to separate touching objects. Default is False.
              exclude_border : Boolean - Remove objects touching the boundaries of the image. Default is false.
    ---------
    Returns
    ---------
             label_masks : Array - Label masks for the input images / trained model / threshold / method.
    """
    # Load images from disk
    images = load_images(im_path, im_folder, file_ext, re_size, new_shape, verbose_mode=False)
    print('Number of imported images: {}.'.format(len(images)))
    # Convert images into tensors for inference
    tensors = image_set_to_tensor_set(images, backbone_prep)
    # Import trained model
    model = import_trained_model_pred(model_name, model_path, l_rate=None, loss_func=None) # No need to have l_rate and loss_func for inference
    # Predict probability maps for the image set
    p_maps = probability_maps_pred(model, tensors)
    # Transform probability maps into label masks
    label_masks = []
    for i in range(0,len(tensors)):
        l_mask = label_mask(p_maps[i].squeeze(), threshold, sq_size, min_size, min_pixel, im_watershed, exclude_border)
        label_masks.append(l_mask)
    print('Number of computed label masks: {}.'.format(len(label_masks)))
    # Save computed label masks
    if save_data == True: 
        filenames = [i for i in os.listdir( im_path + im_folder) if i.endswith(file_ext)] # Load input image file names
        filenames.sort() # Sort input image file names
        for i in range(0,len(label_masks)):
            io.imsave(out_path + out_folder + 'Object_Mask_'+ str(filenames[i]), label_masks[i], plugin='tifffile', check_contrast=False)
        print('Number of label masks saved: {}.'.format(len(label_masks)))
    else:
        pass
    return (images, label_masks)
    
def binary_mask_pred_stitch_stack(im_path, im_folder, file_ext, re_size=False, new_shape=None, backbone_prep=None, model_name=None, model_path=None, patch_size=256, threshold=0.5, sq_size=2, min_size=10, save_data=False, out_path=None, out_folder=None):
    """ Compute and save binary masks for a particular trained model, threshold, and image data set inputed in form of a z-stack.
    ----------
    Arguments
    ----------
              im_path : String - Path to the images to be segmented
              im_folder : String - Name of the folder where the images to be segmented are to be found
              file_ext : String - File extension (e.g. '.tif')
              re_size : Boolean - Resize or not loaded patches
              new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches
              model_name : String - Name of the model to be loaded
              model_path : String - Path to the location where the trained model is saved
              patch_size : Integer - Size in pixels of the patches to be generated prior segmentation and for subsequent tiling
              threshold : Float - Threshold for binary segmentation on [0,1]
              save_masks : Boolean - To save on disk computed binary masks
              output_path : String - Path to the location where the computed binary masks are saved
              output_folder : String - Name of the ouput folder where the computed binary masks are saved
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel
              min_size : Integer - Minimum area in pixel below which a binary element is removed. Default is 10 pixels
              backbone_prep : String - Name of the encoder backbone.
    ---------
    Returns
    ---------
             trim_images : Array - Trimmed images prior segmentation.
             seg_masks : Array - Binary masks for the input images / trained model / threshold.
    """
    # Load z-stack from disk
    images = load_images(im_path, im_folder, file_ext, re_size, new_shape, verbose_mode=False)
    print('Number of imported z-slices: {}.'.format(images[0].shape[2]))
    # Load trained model
    model = import_trained_model_pred(model_name, model_path, l_rate=None, loss_func=None) # No need to have l_rate and loss_func for inference
    # Trim images to an optimal size for patching
    trim_images=[]
    # Take each z-slice and re-slice it to an optimal size for patching
    trim_images=[]
    for i in range (0, images[0].shape[2]):
      t_im = trim_image(images[0][:,:,i], patch_size, patch_size)
      trim_images.append(t_im)
    # Compute binary segmentation masks
    seg_masks=[]
    for i in range(0, len(trim_images)):
      imagesize_h = int(trim_images[i].shape[0]) # Height of the image
      imagesize_w = int(trim_images[i].shape[1]) # width of the image
      patches_im = patchify(trim_images[i],(patch_size, patch_size), step=patch_size)
      #print(patches_im.shape)
      patches_seg_mask=[]
      for j in range(patches_im.shape[0]):
        for k in range(patches_im.shape[1]):
          patch = patches_im[j,k]
          tensor_patch = image_set_to_tensor_set([patch],backbone_prep) # Convert patches into tensors
          p_map_patch = probability_maps_pred(model,tensor_patch) # Use trained model to compute probabilility maps for patches
          b_mask = image_to_binary_mask(p_map_patch.squeeze(), threshold)
          b_mask = binary_mask_postprocessing(b_mask,sq_size, min_size)
          patches_seg_mask.append(b_mask)
      patches_seg_mask = np.array(patches_seg_mask)
      patches_seg_mask_rs = np.resize( patches_seg_mask,(patches_im.shape[0],patches_im.shape[1],patch_size,patch_size))
      binary_mask = unpatchify(patches_seg_mask_rs,trim_images[i].shape) # Generate full scale binary mask
      seg_masks.append(binary_mask)
    print('Number of segmented z-slices: {}.'.format(len(seg_masks)))
    # Concatenate images and mask back into z-stacks
    z_im = np.stack(trim_images, axis=2)
    z_mask = np.stack(seg_masks,axis=2)
    # Save computed binary masks
    if save_data == True:
        filenames = [i for i in os.listdir( im_path + im_folder) if i.endswith(file_ext)] # Load input image file names
        filenames.sort() # Sort input image file names
        io.imsave(out_path + out_folder + 'Binary_Mask_'+ str(filenames[0]), z_mask, plugin='tifffile', check_contrast=False)
        io.imsave(out_path + out_folder + 'Trim_Image_'+ str(filenames[0]), z_im, plugin='tifffile', check_contrast=False)
        print('Number of binary masks saved: {}.'.format(len(seg_masks)))
    else:
        pass
    return (trim_images, seg_masks)


def binary_mask_pred_stitch(im_path, im_folder, file_ext, re_size=False, new_shape=None, backbone_prep=None, model_name=None, model_path=None, patch_size=256, threshold=0.5, sq_size=2, min_size=10, save_data=False, out_path=None, out_folder=None):
    """ Compute and save binary masks for a particular trained model, threshold, and image data set.
    ----------
    Arguments
    ----------
              im_path : String - Path to the images to be segmented
              im_folder : String - Name of the folder where the images to be segmented are to be found
              file_ext : String - File extension (e.g. '.tif')
              re_size : Boolean - Resize or not loaded patches
              new_shape : Tupple - New dimensions (x,y) in pixels of the loaded patches
              model_name : String - Name of the model to be loaded
              model_path : String - Path to the location where the trained model is saved
              patch_size : Integer - Size in pixels of the patches to be generated prior segmentation and for subsequent tiling
              threshold : Float - Threshold for binary segmentation on [0,1]
              save_masks : Boolean - To save on disk computed binary masks
              output_path : String - Path to the location where the computed binary masks are saved
              output_folder : String - Name of the ouput folder where the computed binary masks are saved
              sq_size : Integer - Size of the square structuring element used for morphological opening. Default is 2 pixel
              min_size : Integer - Minimum area in pixel below which a binary element is removed. Default is 10 pixels
              backbone_prep : String - Name of the encoder backbone.
    ---------
    Returns
    ---------
             trim_images : Array - Trimmed images prior segmentation.
             seg_masks : Array - Binary masks for the input images / trained model / threshold.
    """
    # Load images from disk
    images = load_images(im_path, im_folder, file_ext, re_size, new_shape, verbose_mode=False)
    print('Number of imported images: {}.'.format(len(images)))
    # Load trained model
    model = import_trained_model_pred(model_name, model_path, l_rate=None, loss_func=None) # No need to have l_rate and loss_func for inference
    # Trim images to an optimal size for patching
    trim_images=[]
    for i in range(0, len(images)):
      t_im = trim_image(images[i], patch_size, patch_size)
      trim_images.append(t_im)
    # Compute binary segmentation masks
    seg_masks=[]
    for i in range(0, len(trim_images)):
      imagesize_h = int(trim_images[i].shape[0]) # Height of the image
      imagesize_w = int(trim_images[i].shape[1]) # width of the image
      patches_im = patchify(trim_images[i],(patch_size, patch_size), step=patch_size)
      patches_seg_mask=[]
      for j in range(patches_im.shape[0]):
        for k in range(patches_im.shape[1]):
          patch = patches_im[j,k]
          tensor_patch = image_set_to_tensor_set([patch],backbone_prep) # Convert patches into tensors
          p_map_patch = probability_maps_pred(model,tensor_patch) # Use trained model to compute probabilility maps for patches
          b_mask = image_to_binary_mask(p_map_patch.squeeze(), threshold)
          b_mask = binary_mask_postprocessing(b_mask,sq_size, min_size)
          patches_seg_mask.append(b_mask)
      patches_seg_mask = np.array(patches_seg_mask)
      patches_seg_mask_rs = np.resize( patches_seg_mask,(patches_im.shape[0],patches_im.shape[1],patch_size,patch_size))
      binary_mask = unpatchify(patches_seg_mask_rs,trim_images[i].shape)
      seg_masks.append(binary_mask)
    print('Number of computed binary masks: {}.'.format(len(seg_masks)))
    # Save computed binary masks
    if save_data == True:
        filenames = [i for i in os.listdir( im_path + im_folder) if i.endswith(file_ext)] # Load input image file names
        filenames.sort() # Sort input image file names
        for i in range(0,len(seg_masks)):
            io.imsave(out_path + out_folder + 'Mask/' + 'Binary_Mask_'+ str(filenames[i]), seg_masks[i], plugin='tifffile', check_contrast=False)
            io.imsave(out_path + out_folder + 'Image/' + 'Trim_Image_'+ str(filenames[i]), trim_images[i], plugin='tifffile', check_contrast=False)
        print('Number of binary masks saved: {}.'.format(len(seg_masks)))
    else:
        pass
    return (trim_images, seg_masks)







