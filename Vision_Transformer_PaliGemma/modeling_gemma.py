# Paligemma is introduced by google in March 2024 in a blog 
# it is an open-source VLM (vision Language model)

"""Process the text and image that were encoded"""

import torch
from torch import nn 
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math

from modeling_siglip import SiglipVisionConfig, SiglipVisionModel


class KVCache():
    def __init__(self) -> None:
        # KV cache buffer
        self.key_cache: List[torch.Tensor] = []  # we need to append the incoming k and v to this which are of type torch.Tensor
        self.value_cache: List[torch.Tensor] = []  #
        # both have same shape
    
    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        else:
            # key_cache: [b, num_heads_kv, seq_len, head_dim] # from the Gemma Attention block
            return self.key_cache[0].shape[-2]  # how many items in the buffer
    
    def update(  # for the second level in process. for the generation part, we add q value by one token at a time 
        self,
        # add the k and v states to the KV cache of this layer layer_idx
        key_sates: torch.Tensor,  # so, not a list
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if len(self.key_cache) <=layer_idx:
            # if we never added anything to the kv-cache of this layer, we need to create it
            self.key_cache.append(key_sates)
            self.value_cache.append(value_states)
        else:
            # .. otherwise we concatenate the new keys with the existing ones.
            # tensor shape: [b, num_heads_kv, seq_len, head_dim]
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_sates], dim=-2) # along the second last dim; seq_len
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)
        
        # return all the content of the buffer
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


"""
RoFormer: enhanced Transformer with Rotary positional embedding -- introduced RoPE (RPE) 2023
"""
class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        # base: scaling factor
        super().__init__()
        # attn mech performed for each head individually, so, pos encoding is applied to each head
        self.dim = dim # head_dim
        self.max_position_embeddings = max_position_embeddings # max number of positions the model can encode
        self.base = base
        
        # calculate the theta according to the formula
        # theta_i = 10000 ^ (-2*i / d)   # i = 0, 1, 2, ...., d // 2
        # as inverse of -ve sign in power
        """This is different from actua paper, i = 0, 2, 4,6, 8.... dim""" # so, we skip and make it as same size as original
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        
        # just a buffer not considered for model parameters but static
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)
    
    @torch.no_grad()
    def forward(
        self,
        x, # actual vector
        position_ids,  # m - position 
        seq_len=None
    ):  # m - position # given m, calculate the cosine and sine that we need to multiply with vectors
        # x: [b, num_attn_heads, seq_len, head_dim]
        self.inv_freq.to(x.device)
        # copy the inv_freq tensor for the batch in the sequence
        # inv_freq_expaned: [b, head_dim // 2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # [b, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        
        """
        vector * cosine + changed_vector * sine
        """
        with torch.autocast(device_type=device_type, enabled=False):  # for mixed precision
            # multiply each theta by the position (which is the argument of the sin and cos function)
            # freqs: [b, head_dim // 2, 1] @ [b, 1, seq_len] ---> [b, seq_len, head_dim // 2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [b, seq_len, head_dim]
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
"""
# RoPE -- need to understand it better mathematically
"""
def rotate_half(x):
    
    ####### paper
    # for all dim except the last, take all elements; for the last dim, take the first half
    x1 = x[..., : x.shape[-1] // 2] # takes the first half of the last dim
    x2 = x[..., x.shape[-1] // 2 :] # takes the second half of the last dim
    
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # add the head_dim
    sin = sin.unsqueeze(unsqueeze_dim) 
    
    # actual output as per the paper
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed
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

# RMS normalization, refer notes
class GemmaRMSNorm(nn.Module):
    def __init__(
        self, 
        dim: int, 
        eps: float = 1e-6,
    ):
        self.eps = eps
        """Gamma parameter in RMS paper  """
        # weight matrix. as per the dim of the input. one for each dimension.
        self.weight = nn.Parameter(torch.zeros(dim)) # learnable parameter 
    
    # root mean statistics
    def _norm(self, x):  
        # x * 1/sqrt(...)
        # as the ouput should not be very close to 0, as because this will be the denominator. So, we add eps value
        return x * torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)  # RMS formula
    
    def forward(self, x):
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())  # multiply each of the normalized value with the gamma parameter
        
        return output.type_as(x)



#### for repeat the kv to the size of heads in the q for the attention
def repeat_kv(
    # key and value states: [b, head, seq_len, head_dim]
    hidden_sates: torch.Tensor,
    n_rep: int
) -> torch.Tensor:
    
    batch, num_key_value_heads, slen, head_dim = hidden_sates.shape
    if n_rep == 1:
        return hidden_sates
    # make a extra dim that replicates the size of the n_rep. We need to produce n_fold value, so multiply
    hidden_sates = hidden_sates[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)  # make it as 5 dim
    
    # easy peasy ho ho
    return hidden_sates.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaAttention(nn.Module):
    # constructor
    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        # layer_idx: as we will use different KV cache for each layer, in order to know which one should use, layer_idx is used
        super().__init__()
        
        self.config = config
        self.layer_idx = layer_idx
        
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size  # embedding size of each token
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim  # dimension of each of the head.
        
        self.num_key_value_heads = config.num_key_value_heads  # no of heads of the Keys and values
        
        """Grouped query attention: advanced attn mech to improve the efficiency and effectiveness particualarly in large-scale transformer models
        self attn mech: every token attends to every other token. O(n^2 * d) # n - seq len; d - dim of the model 
        
        Grouped Query Attention: divides the queries into groups. Each group attends to a subset of keys, rather than the entire set of keys. Reduces the computation load and memory usage
        """
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # divide the total matrix into many heads and have head level attention and then merge it altogether
        
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_casual = True
        
        # hidden_size = num_heads * head_dim
        assert self.hidden_size % self.num_heads == 0 # should perfectly divide, so, both should be even numbers
        """Different from MHA implemented for SigLip. In SigLip, the input and output of the projection (linear) layer was the same, that is embed_dim
        we are trying to compress the many heads in the K and V into having only few to reduce the copying time from the High Bandwidth Memeory (HBM) and Local Memory 
        
        - For every head in query will take some common heads in Keys and Values. Like 8 heads in Query and 4 in K. So, 2 heads in Q will share 1 head in Key 
        """
        
        # the num of heads for query grouped attention is bigger than the K and V values ---- the no of attention heads of the Q is 8 and for the key_and_value, it is only one
        # EXAMPLE::::: Number of heads = 8; Hidden_Size = 1024; Head_Dim = 1024 / 8 = 128 
        # Wq: [1024, 8 * 128]
        # Wk: [1024, 1 * 1024]  # as the head_dim = hidden_size / num_attention_heads
        # Wv: [1024, 1 * 1024]
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        
        # for converting the output to the original shape
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings = self.max_position_embeddings,
            base = self.rope_theta
        )
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Tuple[torch.FloatTensor]]]:
        
        # In the prefilling, the input will be all the prompt that is Q
        bsz, q_len, _ = hidden_states.size() # [b, seq_len, hidden_states]
        query_states = self.q_proj(hidden_states) # [b, seq_len, num_heads_q * head_dim]
        key_states = self.k_proj(hidden_states) # [b, seq_len, num_heads_kv * head_dim]
        value_states = self.v_proj(hidden_states)
        
        # q_len  -  length of the Q is the length of the hidden_states
        # [b, num_heads_q, seq_len, head_dim]. First view by splitting the hidden_size into num_heads and head_dim. Later, transpose the 1st and 2nd dim
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        """Encoding the position of the states"""
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        
        # positional encoding the queries and keys
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin) # RoPE - based on Relative Positional encoding
        
        # In the second stage in the KV cache, we give only one token in the Q, and this will be added to the KV cache
        # then, we retrieve the K and V from the kv_cache
        
        if kv_cache is not None:
            # sending the key and value states which has the encoded values of the input to be appended to the KV cache
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx) # layer_idx denotes the number of the key state
        
        # kvcache: 2 phases ==> prefilling (Image + prompt) --> token generation
        
        """Don't forget that we are using Grouper query atention, but BUT BUT BUT
        We are reversing the effect of that process. 
        
        """  # we are literally repeating the keys and values states to match the size of the heads in the query. So, this undoes the effect of the Grouped Query attention
        # Repeat the key and values to match the number of heads of the query
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        # perform the calculation Q * K_trans / sqrt(head_dim)  # transpose the last two dimeesntions -- logically and technically
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        assert attention_mask is not None
        """apply attention mask - casual attention mask"""
        attn_weights = attn_weights + attention_mask  # for adding 0's and -inf. and while exponentiatin in softmax, it will be gone
        
        # apply the softmax along the last dimension -- logically
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        # apply the dropout
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training) # dropout only in training
        
        # output of the MHA is multiplied by the values_states
        attn_output = torch.matmul(attn_weights, value_states) # according to the formula
        # so, we have got contextualized tokens at the end. Like each row specifies the aggregated attention of previous and present tokens and like token by token
        
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f" 'attn_output' should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )
        # make it contiguous  # [b, seq_len_Q, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous() # the input and output should of same shape
        # concatenate the head
        # concatenation of output of each head
        attn_output = attn_output.view(bsz, q_len, -1) # (should be 3 dim), [batch, seq_len, hidden_size] # hidden_size = head_dim * num_heads
        # now, each embedding will be the independent calculation, so, we need to merge all the outputs and mingle it. so, we need to pass to Linear layer
        # mixing mechanism
        attn_output = self.o_proj(attn_output) # missing mechanism  # shape of hidden_size
        
        return attn_output, attn_weights
    
    
class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        
        # applies non linearity and up and down projection
        """gate_proj is used by gemma activation function"""
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        
    def forward(self, x):
        # y = self.gate_proj(x) # [b, len, intermediate_size]
        # y = nn.functional.gelu(y, approximate="tanh") # GELU = Gaussian Error Linear Unit - introduces the non-linearity
        
        # j = self.up_proj(y)
        # multiplying the gelu function with the linear projection output
        # z = y * j 
        # z = self.down_proj(z) # [b, len, hidden_size]
        """gate_proj is some learnable projection before sending to the activation layer, gelu"""    
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))
    """
    Gelu output element-wise multiplied with up_proj output. Because, it intorduces a form of gating mechanism, similar to how attention mechanism work in transformer.
    IT'S better TO DO SO
    """
    # gelu is also used in SigLip model. 
    
# Decoder Layer of the transformer
class GemmaDecoderLayer(nn.Module):  # similar to SigLip
    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        
        self.hidden_size = config.hidden_size
        
        # self attention for the imput_embeds which are the combination of the image and text inputs projected toegether
        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        
        # head of the decoder model
        self.mlp = GemmaMLP(config)
        """ the model will have 2 norms as in trans model """
        # 1st one
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 2nd one
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # norm -> attn (add) -> norm -> ffn (add) -> output ->  [[ norm -> linear -> softmax ]] 
        
    def forward(
        self,
        hidden_states: torch.Tensor,  # input to this layer of the transformer
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,   # we need this for Position Encoding, RoPE
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        
        residual = hidden_states
        
        # [b, seq_len, hidden_size]
        # norm 1
        hidden_states = self.input_layernorm(hidden_states)
        
        # Layer 1
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        
        # add the residual connection
        hidden_states = residual + hidden_states
        # [b, seq_len, hidden_size]
        residual = hidden_states
        # norm 2
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        # Layer 2
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states # residual connection - add
        
        return hidden_states


# it is the embedding layer which converts the input embedding of the image and text into the contextual embeddings. That's it and it can be used to produce the text generation output
class GemmaModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__() 
        
        self.config = config
        self.padding_idx = config.pad_token_id  # we are not using pad token
        self.vocab_size = config.vocab_size  # the embeddings depending on the tokens in the voacab 
        # each embedding vector will be of shape of hidden_size
        
        # hidden_size -- size of the embedding vector of the text token
        # Convert the text tokens to embedding with size (vocab_size, hidden_size, padding_idx)
        # Instance, we can use this later to encode to embdding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx) # vocab_size to hidden_size, padding_idx for indicating the padding tokens as it should not produce gradients for the padding_tokens
        
        """We don't apply Positional Encoding after input embedding because we are using RoPE (Rotary Position Encoding)"""
        # List of transformer layers
        self.layers = nn.ModuleList(
            [
                GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # for normalizing and regularization. Different from Layer normalization
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):  # we use this in PaliGemmaForConditionalGenerationModel class to get the embddings initial inputs.
        return self.embed_tokens  # send the embeddings which will be later combined with image embddings.
    
    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,  # apply positional encoding to each token
        inputs_embeds: Optional[torch.Tensor] = None,  # image feature + text tokens
        kv_cache: Optional[KVCache] = None,  # cache
    ) -> torch.FloatTensor:
        
        # [b, seq_len, seq_len, hidden_size]
        hidden_states = inputs_embeds
        # [b, seq_len, hidden_size]
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)  # divide by 2
        # magnitude of the feature should remain the same even if the dimension changes
        hidden_states = hidden_states * normalizer  # normalize 
        
        # DECODER of the transformer model
        for decoder_layer in self.layers:
            # [b, seq_len, hidden_size]
            hidden_states = decoder_layer(  # get the contextualized embddings
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache
            )
        
        # output of last layer
        hidden_states = self.norm(hidden_states)  # RMS normalization
        
        # [b, seq_len, hidden_size]
        return hidden_states
            
# converts the size of the image features (hidden_size) to the size of the embedding size used by the language model. Both should be in same size. 
# resizing
class PaliGemmaMultiModalProjector(nn.Module):  # just a linear layer for the projection. he he 
    def __init__(slef, config: PaliGemmaConfig):
        super().__init__()
        
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)
    
    def forward(self, image_features):
        hidden_states = self.linear(image_features)
        return hidden_states


class GemmaForCausalLM(nn.Module):  # Transformer model + Language modelling head 
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        
        # Transformer model
        self.model = GemmaModel(config)  # so, we need linear layer on top of the LM 
        self.vocab_size = config.vocab_size  # of the LM
        
        # linear layer in the transformer that projects each output embedding into logits and then we use softmax on that logits to get the prediction. Remember that !!
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False) # hidden_size is the output embedding size and we need to convert to the voacb_size to get logits and apply softmax to get the predictions of the output
    
    def get_intput_embeddigns(self): 
        return self.model.embed_tokens
    
    def tie_weights(self):  # share the weights of the embedding layer with the logits layer
        self.lm_head.weight = self.model.embed_tokens.weight  # get the logits from the LM
        # we have done the same earlier. Look at that
    
    def forward(
        self, 
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None
    ) -> Tuple:
        
        # input_ids: [b, seq_len, hidden_size]
        # outputsL [b, seq_len, hidden_size]
        outputs = self.model(   # series of embeddings
            attention_mask = attention_mask,
            position_ids = position_ids,
            input_ids = inputs_embeds,
            kv_cache = kv_cache,
        )
        # turning into logits
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float() # turning to logits
        
        return_data = {
            "logits": logits,
        }
        
        if kv_cache is not None: 
            # return the updated KV cache
            return_data["kv_cache"] = kv_cache
        
        return return_data
        
# This above language model is just a embedding layer which converts the input data into contextual embeddings. that's it 
        
    
                    
        
    
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
        language_model = GemmaForCausalLM(config.text_config)
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
        
        # In KV cache: 1. Prefeeding (send the initial prompt to the model to generate KV cache initially) and 2. Token Generation one at a time  
        # So, in 1st stage, the size of Q, K, V will be the size of the tokens in prompt 
        
        """Create the Attention mask for KV cache """  # to mask out the the forward text. So, Make a attention mask as the same size of the attention matrix. -inf for all the tokens we don't want
        dtype, device = input_embeds.dtype, input_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = input_embeds.shape[1]
        
        # Prefilling. First time 
        if kv_cache is None or kv_cache.num_items() == 0: # initially 
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding 
            """Paligemma working: [image, Prefix, Suffix/target]  << img tokens - bos - text tokens - seperator token - output tokens - eos - pad >>
            
            We don't mask the input tokens (image and text tokens), we only mask the output tokens. 
            - we don't mask image tokens, because, we need to understand all the patches, we don't do that for iamge 
            - we don't mask text tokens / prompt, because it just says what the paligemma should do like the task. so, we need to understand the prompt to give the output. So, we need to understand the future tokens too in order to understand the task.        
            So, no need of mask for the Image and Text tokens. ### we just need to understand the prefix but, we need to generate the suffix, so, we mask the suffix   
            """
            # seperator token = new line \n character to split between the input and the output. We need bos for only the input and eos for only the output 
            # we don't mask anything
            casual_mask = torch.full( # attention mask. """ we are not adding any -inf to the mask. Because, paligemma works differently"""
                # for the prefilling, the Q matrix should be in the shape of number of input tokens
                size=(batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device  # for the prefilling 
            )
        else:  # we don't mask anything even in 2nd stage as we want only the last row of the attention mask. So, no need to mask 
            # But, during training, we will have the causual mask, but, in inference, we won't have that 
            # later stages
            # since, we are generating tokens, the query must be one single token. One at a time 
            assert q_len == 1 
            kv_len = kv_cache.num_items() + q_len  # we need to add the new token to he KV cache 
            # also, in this case we don't need to mask anything, since each query should be able to attend all previous tokens 
            # This only works when we have no padding 
            casual_mask = torch.full(  # as token by token, only one query, q_len == 1 
                size=(batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )
        
        # Add the head dimension   # one attention computation for each head, so, attention_head will be 1. 
        # We will have one head for each of the token generation, so, num_heads_q = 1  
        # [b, q_len, kv_len ] --> [b, num_heads_Q, q_len, kv_len]
        casual_mask = casual_mask.unsqueeze(1)  # add a extra dim in the first dimension
        
        # for training, we need attention_mask 
        """For positional encoding"""  # to maintain the order of the tokens. 
        # for the 2nd stage 
        if kv_cache is not None and kv_cache.num_items() > 0:
            # the position of the query is just the last position 
            position_ids = attention_mask.cumsum(-1)[:, -1]  # last dimenstion (-1) and selects the last value from the cumulative sum, which is the position of the current token (token being generated)
            
            # check if the position_ids is the 1D tensor which happens if there is only one token in the batch
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)    # add an extra dim, [batch_size, 1]
        
        # KV cache is empty or means the 1st iteration 
        else:
            # create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position 
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
            
        
        return final_embedding, casual_mask, position_ids
     
        
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
        
        
        