# Paligemma is introduced by google in March 2024 in a blog 
# it is an open-source VLM (vision Language model)

"""Process the text and image that were encoded"""

import torch
from torch import nn 
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class GemmaConfig(): # config of overall LLM 
    def __init__(
        self, 
        vocab_size, # how much tokens we have in vocab 
        hidden_size,  # size of the embedding vector of each token  
        intermediate_size, # intermediate layer of Feed forward layer 
        num_hidden_layers,  # how many layers of Transformer stack
        
        # Grouper Query Attention # different no of heads of Q, K and V  
        num_attention_heads,  # for queries 
        num_key_value_heads,  # for K and V 
        head_dim=256, # dim of each head   #         heads * head_dim = embed_dim 
        max_position_embeddings=8192,  # max number of tokens in the sequence
        rms_norm_eps=1e-6,   # RMS norm epsilon
        # Rotary positional Encoding
        rope_theta=10000.0,
        attention_bias=False, # bias for attention matrix 
        attention_dropout=0.0,   # 
        pad_token_id=None, 
        **kwargs,     
    ):
        
        super().__init__()
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id
        
    
    
# structure of the model and then components of the model
# Paligemma with Siglip vision encoder and Gemma Language model 
class PaliGemmaConfig():
    def __init__(
        self, 
        vision_config = None,   # config of the vision encoder
        text_config = None,     # config of the text encoder 
        ignore_index = -100,    #### Only for training 
        image_token_index = 256000,  # token corresponding to the placeholder <image> token
        vocab_size = 257152, 
        projection_dim = 2048,  # final dim to which image feature should be resized to  # output size of the projection layer 
        hidden_size = 2048,  # embedin size of the language model 
        pad_token_id = None, 
        **kwargs,
    ):
        
        super().__init__()
        
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        
        # {"param1": value1, "param2": value2, .....}   unpacking 
        self.vision_config = SiglipVisionConfig(**vision_config) # unpacking dict into keyword arguments
        self.text_config = text_config # create an instance 
                
        self.is_encoder_decoder = False   # for the use of HuggingFace, we won't use this
        
        # text config 
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size
        
        # each patch size = 16, so, to calculate the number of patches, image_size // patch_size. Because, each patch has one embedding of dim embed_dim. 
        # to make the enough place holders for the <image> in the text embeddings. # how many patches fit along one dimension of the image ** 2 
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2 # (224 // 16) ** 2 = 14 ** 2 = 196 
        # image tokens each image will generate -- how many patches per image 
        self.vision_config.projection_dim = projection_dim # to resize the image tokens to match the text tokens
        
        
class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: PaliGemmaConfig):
        self.config = config
        
        # encoder of the image 
        self.vision_tower = SiglipVisionModel(config.vision_config)
        
        # Linear projection that will convert the size of embedding output by vision encoder into the size of embedding of text token. 
        # So, that can be concatenated together. 
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size 
        
        # Transformer Decoder 
        language_model = GemmaForCasualLM(config.text_config)
        self.language_model = language_model
        
        # for pad_token_id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1 
    
    def tie_weights(self):
        """
        most LLM are Decoder only model. However, they don't use the MHA, instead they use Masked-MHA and the block repeated for N times. 
        # The input to the decoder is given by "Input Embedding" which converts the vocab ids into embedding of needed dim. 
        # And, then after N layers of decoder, at the last stage, the "Linear" layer converts the embeddings into vocab size. 
    
    So, Both are doing the inverse functions. Just opposite to each other. So, we can use the learned parameters from the "Input Embedding" on the "Linear" layer. 
    
    This method is known as WEIGHT TIE --- share the parameters 
        """
        # imlemented in GemmaForCasualLM
        return self.language_model.tie_weights() # reusing the parameter of one layer into another 

    
    def _merge_input_ids_with_image_features(
        self, 
        image_features: torch.Tensor,   # extracted image feature 
        input_embeds: torch.Tensor,   # embedding for text features
        input_ids: torch.Tensor,   # ids of the input features
        attention_mask: torch.Tensor,   # for attention mask.. to avoid looking into forward or later words 
        kv_cache: Optional[KVCache] = None,  # regularization
    ):  # both image and text features should have the same size of "embed_dim"
        # embed_dim -- sent after projection. so, same size of text tokens
        _, _, embed_dim = image_features.shape  # [b, seq_len, embed_dim]  # seq_len or path_size 
        # input_ids - numbers indicating the position of each token in the vocab 
        batch_size, sequence_length = input_ids.shape   # how many tokens we have in total including the image and text tokens  # overall shape  
        # sequence_length - no of input ids we have -- for the text tokens
        dtype, device = input_embeds.dtype, input_embeds.device    # type and device of the input_embeds 
        
        # Shape: [b, seq_len, hidden_size] # scale the iamge features after projection
        scaled_image_features = image_features / (self.config.hidden_size ** 0.5) # divide by sqrt(hidden_size)  # like in attention mechanism
        
        # combine the mebddings of the iamge tokens, the text tokens and mask out all the padding tokens   # of size of image features   
        final_embedding = torch.zeros(batch_size, sequence_length, embed_dim, dtype=input_embeds.dtype, device=input_embeds.device) # sequence_length of empty embeddings of size embed_dim which is same size of the image embed_dim
        
        """Masking""" # where to put the embeddings of image, text, padding tokens in the "final_embedding"
        # [b, seq_len] # just for the seq_len,. we need to mask out image and pad tokens 
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)  # not a image_token and not a pad_token_id  # [0, 0, 0, 0, 1, 1, 1, 1, ,,,] # for test token  # 1 - true; 0 - false
        # [batch, seq_len]. True for image tokens
        image_mask = input_ids == self.config.image_token_index  # only the image tokens. equals to image placeholder tokens 
        # [batch, seq_len] - true for padding tokens
        pad_mask = input_ids == self.pad_token_id   # only the pad_token_id
        
        # EXPAND the masks to the embedding_dim otherwise we can't use them in torch.where
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)  # repeating the 0s and 1s along new dimensions
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        """putting altogher"""
        # for including the text 
        # Add the text embeddings   # if condition (non-zero, i.e, 1) # take input_embeds and when false, take the final_embeddings # if condition (non-zero, i.e   
        final_embedding = torch.where(text_mask_expanded, input_embeds, final_embedding)
        # for including the image   # can't use "where" because the scale_image size is not equal to final embedding 
        final_embedding = torch.masked_scatter(image_mask_expanded, scaled_image_features) # copy from the scaled_image_features for where the image_mask is true 
        # zero out paddings, we don't care that literally 
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding) # if padding, put zero in that, else, as it is 
        
        
        
    def forward(
        self, 
        input_ids: torch.LongTensor = None, 
        pixel_values: torch.FloatTensor = None,  # rescaled, resized, normalized image from the Paligemma processor 
        attention_mask: Optional[torch.Tensor] = None,  # as we are not using padding, the attention_mask will be series of 1's 
        kv_cache: Optional[KVCache] = None, 
    ) -> Tuple:
        
        # accessing all the elements of the attention_mask 
        assert torch.all(attention_mask == 1), "The input cannot be padded"
        
        # 1. Extra the input embeddings 
        # convert the {image_toke}{bos}{prompt_token}{SEP}  into embeddings and get the input_ids 
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids) # (b, seq_len, hidden_size)
        
        # 2. Merge text and iamges   
        # forward method: (b, channels, height, widht) ---> (b, num_patches, embed_dim)
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype)) # feed the pixel values to the vision tower with same data type of the input_embeds
        # resize these embeddings into the same size of the embeddings of the language model 
        # [b, patches, embed_dim] -> [b, patches, hidden_size]
        image_feature = self.multi_modal_projector(selected_image_feature)  # projecting the image feature into the size of the expected from the text embeddings 
        # merge both text and image embeddings. We already have place holders in the text embeddings "<image" to place the image embeddings 
        # replace <image> tokens from the text embeddings with the image embeddings
        """ ** merge the text and image feature space."""
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(  ## KV Cache 
            image_feature,  # extracted from the image encoder
            inputs_embeds,  # extracted from the text encoder -- has text tokens and the placeholders 
            input_ids, 
            attention_mask,
            kv_cache
        )
        # then the Transformer Decoder will use the overall embedding which has image embeddings and text embeddings and then produce the output 
        
        # get the output from the Transformer Decoder
        outputs = self.language_model(
            attention_mask=attention_mask, 
            position_ids=position_ids, 
            inputs_embeds=inputs_embeds, 
            kv_cache=kv_cache,
        )
        
        return outputs 
        
        
        