"""Contrastive Vision encoder. -- Signal to latent-Image pretraining """
# as we need encodings of iamge and text to represent and each other and to have same dimensions of the encoding.   
# so, before sending to the language model, we need to send that through the Linear Projection, to make that of the same size. 
from typing import Optional
from typing import Tuple 

# Vision transformer is trained in a way called "Contrastive Learning" like CLIP, SigLip
"""
Contrastive Learning:
    - Type of self-supervised learning. Learn representation by bringing positive examples closer together, and pushing dissimilar negative examples apart. 
    - Anchor, Positive, Negative 
    
    * Embedding space: model learns to project input data into a high-dim embedding space 
    * Loss function: "Constrastive" loss" and "triplet loss"    

"""

import torch
import torch.nn as nn 

# CLIP -- by OpenAI  (Contrastive Language-Image Pretraining)
"""
    Learns to connect images and text by training on a vast dataset of iamge-caption pairs. Uses Contrastive Learning    
    Uses two models, Image model like ViT and another model for Text
    - zero short learning 
    """
# SigLip -- Signal-to-Latent-Latent-to-Image-Pretraining
"""
    extends constrastive learning, often in multimodal contexts, text, images, signal are aligned in a shared latent space. 
    
    - signal to latent 
    - latent to image 
    - contrastive learning 
    
    """

class SiglipVisionConfig:
    def __init__(
        self, 
        hidden_size=768,   # size of the embedding vector of the ViT  # embed_dim 
        intermediate_size=3072, # size of the linear layer in the FFN 
        num_hidden_layers=12,  # no of layers of the ViT 
        num_attention_heads=12,  # in multi-head attention
        num_channels=3,  
        image_size=224,  # 224, 414, 896 # paligemma size 
        patch_size=16,  # size of each patches in pixel
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        num_image_tokens: int = None,   # how many output embeddings the ViT will generate. how many image embedding we will have for each image
        **kwargs
    ): # image encoder converts image to one single embeddings
        
        # ViT outputs Contextualized embeddings. 
        # not one single embedding that represents the whole image but, list of embeddings that represent the patch of the image but also information of the correlation to other patches with attention mechanism 
        
        super().__init__()
        
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.image_size = image_size
        self.patch_size = patch_size
        self.layer_norm_eps = layer_norm_eps
        self.attention_dropout = attention_dropout
        self.num_image_tokens = num_image_tokens

class SiglipVisionEmbedding(nn.Module):
    # extract info from the patches 
    
    def __init__(self, config: SiglipVisionConfig):
        self.config = config 
        
        self.embed_dim = config.hidden_size # embedding dimensions of each of the patches that will be encoded 
        self.image_size = config.image_size
        # size of each patch. 
        # each patch will have a size of 16 x 16 pixels 
        self.patch_size = config.patch_size 
        
        # notice how the patches are divided, it is not actually divided, but then, it is processed in a way that each are independent 
        # extract information from each patch independently 
        self.patch_embedding = nn.Conv2d(  # for each output channel, we have one kernel 
            in_channels=config.num_channels, 
            out_channels=self.embed_dim, # refers to the dim of each patch 
            kernel_size=self.patch_size,  
            stride=self.patch_size, # without overlap 
            padding="valid",   # no padding is added, the conv operation will only consider valid patches within the original image dim
        )
        
        # 224 / 16 = 14 
        self.num_patches = (self.image_size // self.patch_size) ** 2   # like, num of patches means, the row and the column together, so, it would be x * x
        self.num_positions = self.num_patches # we need these many positional embeddings for our image 
        # pos embedding is a vector of same size of the patch. How many positions and dim of each position 
        self.positional_embeddings = nn.Embedding(num_embeddings=self.num_positions, 
                                                  embedding_dim=self.embed_dim) # create the embeddings of all the patches and each in the size of the embed_dim, 768 
        
        # register a tensor as a buffer within an nn.Module 
        # this means that the tensor is part of the model's state but isn't considered a parameter to be learned or updated through backprop 
        # Buffers can store things like running averages, fixed positional encodings, or other state information needed during training 
        self.register_buffer(
            "position_ids",  # name of the buffer being registered
            torch.arange(self.num_positions).expand((1, -1)), # positional indices. static. expand((1, -1)) - new view of the tensor with a different shape. # add extra 1st dim and let pytorch infer the last dim 
            persistent=False # won't be save with the model when calling model.save_state_dict(). 
        )
    
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        _, _, height, width = pixel_values.shape # [b, c, h, w]

        # embed the information from the image to a embedding vector with conv2d. 
        # 768 channels, so, 768 kernels, and H x W spatial dimensions
        patch_embeds = self.patch_embedding(pixel_values) # output from conv2d: [b, embed_dim, num_patches_H, num_patches_W] 
        # each img will be list of embeddings. 
        # [b, embed_dim, num_patches_H, num_patches_W] -> [b, embed_dim, num_patches] 
        embeddings = patch_embeds.flatten(2)  # means, dim starting from 2nd index  # flattens the last two dimentsion into a single dim, representing the total number of patches 
        # element-wise 
        embeddings = embeddings.transpose(1, 2)   # [b, num_patches, embed_dim] # makes sense, you know 
        # register_buffer - num_positions --> sent to embeddings, which has embedding_dim of 768
        
        embeddings = embeddings + self.positional_embeddings(self.position_ids) # getting from the buffer 
        
        return embeddings
        

# MHA is the way for contextualizing stuff 
class SiglipAttention(nn.Module):
    """
    Input: X - tensor of all patches embeddings as row vector: this has information about itseelf only 
    output X - tensor of all patches embeddings as row vector: but, this has information about every other patch relation to each of the patch embedidngs. 
                how each patch is related to each other.  
    """
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads  # head_dim 
        self.scale = self.head_dim ** -0.5 # 1 / sqrt(head_dim)
        self.dropout = config.attention_dropout
        
        # the size of W_q, W_k, W_v are the (embed_dim x embed_dim ) --> then it is done as (embed_dim, num_head, head_dim)
        # 2nd group is divided into number of heads of size head_dim 
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)  # in_channels and out_channels are the same 
        
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
    
    def forward(
        self, 
        hidden_states: torch.Tensor, 
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # hidden_states: b, num_patches, embed_dim  # each of the patch is represented by a vector with shape embed_dim
        batch_size, seq_len, _ = hidden_states.size() 
        """from X to Q, K, V"""
        # first, we run the inputs through 3 transformations 
        # Paramater matrices. No contextual happeining here. 
        # self attention mech needs to see the sane sequence in three different ways as Q, K, V
        """(b, seq_len, embed_dim) -> (b, seq_len, embed_dim, embed_dim)""" # we have to view this as (b, seq_len, num_heads, head_dim)
        query_states = self.q_proj(hidden_states) # [b. num_patches, embed_dim] 
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        """splitting the embed_dim into smaller parts called head_dim with num_heads """
        # split these into smaller tokens based on how many heads we have 
        """
        [b, seq_len, embed_dim, embed_dim] --> [b, seq_len, num_heads, head_dim] and again this is transposed to another -> [b, num_head, seq_len, head_dim]
        """  # embed_dim = num_heads * head_dim 
        # b, num_patches num_heads, head_dim --------> num_heads, b, num_heads, num_patches, head_dim   # seq_len = num_patches
        #### TRANSPOSE: (seq_len, num_heads, head_dim) ----> (num_heads, seq_len, head_dim)... all the stuff in one seq_len will be in one head as one element. and withing that (each of the seq_len, head_dim), there will be head_dim elements 
        # simply, each of the 1st dim (head) will have all the head_dim across all the seq_len. ===== so, could be parallelized 
        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).translate(1, 2)  ####### PERFECT 
        key_states = key_states.view(batch_size, seq_len, self.num_heads, self.head_dim).translate(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_heads, self.head_dim).translate(1, 2)
        #(batch, num_heads, seq_len, head_dim)       
        # Q = X * W_q 
        # K = X * W_k
        # V = X * W_v
        
        # each head work with smaller embedding subset. so, num_heads = 12 (dfined). so, the each of the head will watch only the the subset, so called head_dim. 
        """
        Word may have different meanings according to the context. For example, a word can be verb as well as verb. So, some head will learn it as noun and other will learn it as verb. So, it will get the full understanding. 
        # As all the heads work independently, the training could be parallelized. 12 heads could be in 12 GPUs. 
        # each head will be the dot product of all head_dim elements. 
        """
        # Q_head_1 :     each head with be (seq_len, head_dim)  ----- seq_len times of row_vectors of size head_dim. 
        # K transpose: it becaomes the column vectors, that's why we change the seq_len, head_dim  
        # (batches, num_heads, seq_len, seq_len) # square matrix   # multiple square matrices as per the num_heads 
        attn_weights = (torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale)    # (seq_len, head_dim) x (head_dim, seq_len) ---> representes the dot product of one token with every other token. ROWS: (Q), COLUMNS: (K)
        # we are simply relating one token with every other token. and we call that Q and K    # relationships between two tokens. # denser the result, more the relationships 
        
        # this matrix is calculated independently for each head. computed parallely. 
        """ we apply attention mask. because we want each and every token should relate to every other token preceding that. It should not be related to the future tokens. -- autoregressive model. so, we mask upper triangular matrix with -inf or 0 -- because, we apply softmax (exp ^ a / sum(exp ^ a)) - exp ^ -inf is 0. """
        # WHY WE USE MHA: 
        # 1. we wantt o parallelize the computation 
        # 2. each head should learn to relate tokens or patches differently 
        
        if attn_weights.size() != (batch_size, self.num_heads, seq_len, seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, seq_len, seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # apply softmax along the last dimension--- every square matrix  
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float).to(query_states.dtype)
        # apply dropout only during training  - to reduce overfitting -- regularization
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        # multiply the attn_weights by the value states
        
        """(seq_len, seq_len) x (seq_len, head_dim) -> (seq_len, head_dim)"""
        attn_output = torch.matmul(attn_weights, value_states) # [b, num_heads, seq_len, seq_len] # seq_len is same as num_patches 
        # attention mask says how each token contirbute or the importance for the next prediction. 
        # as softmax is done across the attn_weigths, this will have the importance prob distribution
        """we need to merge all the attn matrices computed across each of the head ---- concat all the heads attn matrices and then multiply by the W_o proj matrix of size ()"""
        # 5. TRANSPOSE: (num_heads, seq_len, head_dim) --> (seq_len, num_heads, head_dim) :::: make it as a sequence as each of the head looks into only the subset of the original matrix. 
        # for example, head_dim = 128 and num_heads = 8 ::: ROW1: [1....128] [129....256]... [...1024] from each of the head. so, (seq_len, num_heads, head_dim)
        
        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f" attn_output should be of size {(batch_size, self.num_heads, seq_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        
        # transpose back: (b, heads, seq_len, head_dim) -> (b, seq_len, heads, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous() # after transposing the tensor, the memory layour might not be contiguous. Non-contiguous tensors can be more difficult to work with because some operations expent tensors to be stored contiguously in memory. 
        # for efficiency and compatibility 
        # (b, seq_len, heads, head_dim) -> (b, seq_len, embed_dim)  # embed_dim = num_heads * head_dim 
        attn_output = attn_output.reshape(batch_size, seq_len, self.embed_dim) # as one matrix where each row (seq_len) represents the whole whole embed_dim. how each patch represent the overall other dim 
        
        """input is (seq_len, embed_dim) and output is also the same """
        # we multiply by W_o to correlate each of the head output as each of the head is computed parallely, none of them are related and to relate them, we do this.    
        # (seq_len, embed_dim) x (embed_dim, embed_dim) = (seq_len, embed_dim)
        attn_output = self.out_proj(attn_output)
        
        # we return both the attn_output and the attn_weights
        return attn_output, attn_weights 
    
class SiglipMLP(nn.module):
    
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config = config 
        
        # again to the same input dimension 
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size) # intermediate size could be 3 or 4 times the input size 
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states) # first layer
        # non linear functions 
        hidden_states = nn.functional.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states) # second layer
        return hidden_states
        
 
class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.embed_dim = config.hidden_size
        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)                                                              
    
    def forward(
        self, 
        hidden_states: torch.Tensor,  # patch embedding with position encoding |or| the output from the previous layer of the decoder 
    ) -> torch.Tensor:
        # residual: b, num_patches, embed_dim 
        residual = hidden_states # to be added with the result 
        
        # Layer norm:
        hidden_states = self.layer_norm1(hidden_states)  # first layer 
        # give contextualized embeddings 
        hidden_states, _ = self.self_attn(hidden_states=hidden_states) # does not change the shape of the input tensor  
        # residual connections from the input to after the layer norm and the self attention
        hidden_states = residual + hidden_states 
        
        # store the hidden states as again we have to make a skip connections 
        residual = hidden_states
        # Layer norm2:
        hidden_states = self.layer_norm2(hidden_states) # second layer norm 
        # input embedding, transforms independently from each other
        #  MLP: as independently from each other, it has more degree of freedom to learn, 
        #       It helps the contextualized embeddings from this layer to get ready for the next layer    
        #       helps introduce non-linearity. helps to learn more complex context 
        hidden_states = self.mlp(hidden_states) # MLP head. series of linear layer.
                
        hidden_states = residual + hidden_states
        
        return hidden_states  # [b, num_patches, embed_dim]
    # doesn't change the dimension of the embedding in the Encoder   
        
        
class SiglipEncoder(nn.Module):
    def __init__(self, config: SiglipVisionConfig):
        super().__init__()
        
        self.config =  config
        
        # stack upto the num of hidden_layers 
        self.layers = nn.ModuleList( # output of previous layer is the input of the next layer 
            [
                SiglipEncoderLayer(config) for _ in range(config.num_hidden_layers)
            ]
        )
    
    def forward(
        self, 
        input_embeds: torch.Tensor
    ) -> torch.Tensor:
        # [batch_size, num_patches, embed_dim] # num_patches = seq_len
        hidden_states = input_embeds
        
        for encoder_layer in self.layers:
            hidden_states = encoder_layer(hidden_states) # this will output the att_output and the attn_weights 
        # no change in the dimension of the layers between stack of encoder layers 
        return hidden_states # return both the outputs   
    
class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: SiglipVisionConfig): 
        super().__init__() 
        
        self.config = config 
        embed_dim = config.hidden_size
        
        # extract patches from the image 
        self.embeddings = SiglipVisionEmbedding(config) # extract and correlate the information of the patches on to the embedding 
        # send those patches to the Encoder 
        self.encoder = SiglipEncoder(config)   # stacks of multiple encoder layers 
        
        """Covariate Shift"""
        # "Covariate Shift". the change in input distribution in batches would greatly affect the output and therefore the loss, similarly big change in gradient would result in big change in weights of the network 
        # So, we normalize - batch normalize (not good) - if the distribution changes for each batches, the mean and SD would not represent every batches equally
        self.post_layernorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps) # for regularization and converging faster and stable training 
        # so, we calculate those statistics along item dim not along batch dim 
    # pixel_values - batches of the image 
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # pixel_values: [b, c, h, w] -> [b, p, d] # p - num_patches, d - embed_dim 
        
        # extracting patches from the image. (Run the conv, and flatten)
        hidden_states = self.embeddings(pixel_values)  # embed the image to different patches and return that. 
        
        # this will be sent to the Encoder to get the contextualized embeddings 
        last_hidden_state = self.encoder(input_embeds=hidden_states) # pass the image to the encoder 
        # contextualized embeddings would have the information about the context and how it is related to other patches with attention mech
        # A = softmax(QK_t / sqrt(d_k)) * V
        
        last_hidden_state = self.post_layernorm(last_hidden_state) # layer norm that contextualized embeddings
        
        return last_hidden_state
    

"""For the contrastive learning, 
we need single embedding for each of the image in the training set, remember the matrix, rows of images and columns of text output which we used to do the contrastive learning from the training set 

# One single embedding for each image, BUT SO FAR, we have generated sequence of contextualized embeddings

"""
class SiglipVisionModel(nn.Module):
    def __init__(self, config: SiglipVisionConfig):  # pass the instance of the config class 
        super().__init__() 
        
        self.config = config # of type of the class 
        self.vision_model = SiglipVisionTransformer(config) # instead of passing the parameters, just pass one class config to every each other
    
    # pixel_values - raw_image data corresponding to the pixel of the image.
    # this will get processed later on 
    def forward(self, pixel_values) -> Tuple:
        # [b, c, h, w] -> [b, num_patches, embed_dim]
        return self.vision_model(pixel_values=pixel_values) # returns the embedding for each patch, so that's why.. [patch, embed_dim]
    
    
###### Contrastive Vision Encoder is done...  
# tokenize the text ---> combine the image tokens with the text tokens. 
