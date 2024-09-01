"""Numpy-like pseudocode for the core of an implementation of CLIP"""
# image_encoder - ResNet or Vistion Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] # batch, h, w, c = minibatch of aligned images 
# T[n, i] # batch, len = minibatch of aligned texts
# W_i [d_i, d_e] - learned proj of image to embed
# W_t [d_t, d_e] - learned proj of text to embed 
# t - learne temperature parameter

""" d_e - dim of the shared embedding space """
import numpy as np 
from torch import nn 
import torch.nn.functional as F 

I = None # image tensor
T = None # text tensor
t = None # temperature parameter - sharpnes of the similarity distribution 
n = None # number of pairs in batch -  batch size 
def image_encoder(I): pass
def text_encoder(T): pass

W_i = None
W_t = None

"""
d_i and d_t do not have to be the same 
# d_i - dim of feature vector produced by image encoder
# d_t - dim of feature vector produced by text encoder 

- To use these different feature vectors in a joint multimodal setting, we need to map both image and text features into a shared embedding space of the same dim d_e. 
This is done suing learned projection matrices. 

W_i and W_t --- same no of columns, but different number of rows.
"""     
# extract feature representations from image and text
# convert a list of images and texts into a list of embeddings 
I_f = image_encoder(I)  # [n, d_i]   
T_f = text_encoder(T) # [n, d_t]

# joint multimodal embedding [n, d_e]
# make sure both image and text embeddings have the same number of dim and then normalize the vectors 
"""
when dot product.. result would be [n, d_e] <-- [n, d_i] x [d_i, d_e] , same for text 

so, it is projected into the shared embedding spae with the W_i and W_t 
then normalized to have unit length
"""
I_e = F.normalize(np.dot(I_f, W_i), dim=1)    # along column, last dimension  

T_e = F.normalize(np.dot(T_f, W_t), dim=1)

# computes pairwise cosine similarities between all paris of the image and text, so it will n x n 
# [n, d_e] x [d_e, n] = [n, n]
# scaled pairwise cosine similarities [n, n]
# 2D tensor # Computing all possible dot products 
logits = np.dot(I_e, T_e.T) * np.exp(t)  # transpose the Text embeddings before dot product 
# forms a matric we need. 
# diagonal elements should have the maximum value 
criterion = nn.CrossEntropyLoss()

# symmetric loss function # 1D tensor 
# 0th image paired with 0th text and so on, that's why we have [0 to n-1]
labels = np.arrange(n) # tensor table [0, 1, 2, .. n-1]

# Teach the model which item in each row/column needs to be maximized 
# similarity of I_x with all texts
loss_i = criterion(logits, labels, axis=0)  # which in row = Image given text 
# similarity of T_x with all images 
loss_t = criterion(logits, labels, axis=1)  # which in column = Text given image 

# total loss 
loss = (loss_i + loss_t) / 2 