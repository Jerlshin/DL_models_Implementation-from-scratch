import torch
import torch.nn as nn

# self-attention 
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        # d_k - dimension, embed_size
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        # embed_size = heads * head_dim
        assert (self.head_dim * heads == embed_size) # embed size needs to be divisible by heads

        self.values = nn.Linear(embed_size, embed_size)
        self.keys = nn.Linear(embed_size, embed_size)
        self.queries = nn.Linear(embed_size, embed_size)
        
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    ''' Learning based Positional Embedding'''
# importance of different parts of the input sequence when making predictions

    # calculating the attention scores, applying mask if needed and softmax to get the attention weights
    def forward(self, values, keys, query, mask):
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]


        # after reshape, we ahve to pass it through Linear layers
        values = self.values(values)  # (N, value_len, embed_size)
        keys = self.keys(keys)  # (N, key_len, embed_size)
        queries = self.queries(query)  # (N, query_len, embed_size)

        # split embedding into self.heads blocks
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)

        # (queries.shape, keys.shape) --> energy.shape
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) # [(N, query, heads, head_dim), (N, key_len, head, head_dim)] --> (N, heads, query_len, key_len)

        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        ### energy shape: (N, heads, query_len, key_len) 
        '''energy = Q * K.T'''
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20")) # upper triangular matrix

        '''Attention(Q, K, V) = softmax(energy / (sqrt(d_k)))'''
        attention = torch.softmax( energy / (self.embed_size ** (1/2)), dim=3)

        # key_len and valu_len matches | # l - for the dimension we want to multiply across
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, heads, heads_dim)
        ### (N, query_len, heads, head_dim) -> then flatten

        out = self.fc_out(out)

        return out

'''Sinusoidal Positional Encoding -- used in the paper mam'''

import numpy as np
# to calculate the possible angles for the sine and cosine positional encodings
def get_angles(pos, i, d):
    # pos - column vector for positions, pos of elements in the sequence
    # i - row vector, dim (indices) of the pos encoding
    # d - int, encoding size, no of dim in the position encoding
    angles = pos / np.power(10000, (2 * (i//2)) / np.float32(d))
    return angles

import tensorflow as tf
# these positional encodings are added to the input embedding of the sequence to give the model information about the order of the element
def positional_encoding(positions, d): # max no of positions to be encoded, # encoding size
    # matrix of all the angles
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
    # sine to even indices in the array, 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) # select all rows and every second column starting from teh first column
    # cosine to odd indices in the array 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = angle_rads[np.newaxis, ...] # adds extra dim to the angle_rads, for batch dim, 2D matrix to 3D tensor
    
    return tf.cast(pos_encoding, dtype=tf.float32) # used to change the data stype of a tensor

'''This methods has disadvantage of fixed pattern'''
# pos_encoding = positional_endcoding(x, y) 


# in paper, transformer block has self-attention and feed-forward blocks with add and norm 
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        
        super(TransformerBlock, self).__init__()
        
        self.attention = SelfAttention(embed_size, heads)
        
        # Batchnorm - average acorss the whole batch
        # Layernorm - average for each layer - more computation
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        '''This does not change the dim, it maps back to the same input dim'''
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        # computing the self-attention
        attention = self.attention(value, key, query, mask)

        # adding the result to the input - skip conection as we did in ResNet 
        # skip connection, run through normalization and finally dropout
        x = self.dropout(self.norm1(attention + query))
        # feed forward 
        forward = self.feed_forward(x)
        # dropout with add and norm
        out = self.dropout(self.norm2(forward + x))
        return out


"""Encoder Block"""

# for processing the input sequence
class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device,
                 forward_expansion, dropout, max_length): # max_lenght - for positional encoding
        
        super(Encoder, self).__init__()
        
        self.embed_size = embed_size
        self.device = device
        
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    # adding positional embedding --> multiple layers of transformer block --> encoded representation
    def forward(self, x, mask):
        N, seq_length = x.shape

        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        # positions = positional_encoding(self.max_length, self.embed_size).to(self.device)
        
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask) # V, K, Q all are same
        
        return out

"""Decoder Block"""

# with masked-self attention --> adding the result to the input, -> add and norm --> feed forward
class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.attention = SelfAttention(embed_size, heads=heads)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, trg_mask):
        # masked multi-head attention
        attention = self.attention(x, x, x, trg_mask)
        
        # taking only the query
        query = self.dropout(self.norm(attention + x)) # Add and norm
        out = self.transformer_block(value, key, query, src_mask)
        return out

# for generating the output sequence
class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers,
        heads, forward_expansion, dropout, device, max_length):

        super(Decoder, self).__init__()
        
        self.device = device
        
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    # process the target sequence by adding positional embedding --> decoder blocks --> output logits
    def forward(self, x, enc_out, src_mask, trg_mask):
        N, seq_length = x.shape
        
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out
    

''' Main Block '''
# bringin together the encoder and decoder
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, 
                 embed_size=256, num_layer=6, forward_expansion=4,
                 heads=8, dropout=0, device="cpu", max_length=100):
        
        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size=src_vocab_size,
            embed_size=embed_size,
            num_layers=num_layer,
            heads=heads,
            device=device,
            forward_expansion=forward_expansion,
            dropout=dropout,
            max_length=max_length
        )

        self.decoder = Decoder(
            trg_vocab_size=trg_vocab_size,
            embed_size=embed_size,
            num_layers=num_layer,
            dropout=dropout,
            heads=heads,
            forward_expansion=forward_expansion,
            device=device,
            max_length=max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.device = device
    
    # mask for the source sequence to ignoise the padding tokens during attention
    def make_src_mask(self, src): 
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)         # (N, 1, 1, src_len)

        return src_mask.to(self.device)

    # casual mask for the target seq to enforce the order of the generation
    def make_trg_mask(self, trg): # traiangular matrix
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones(size=(trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len  # Batch_size, 1, heigth, width
        )    

        return trg_mask.to(self.device)
    
    # simple implem
    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)

        return out
    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)

## Sinusoidal based positional encoding -- for low data, we can use this mam. but it has fixed encoding

'''
import torch
import torch.nn as nn

def positional_encoding(max_length, embed_size):
    position = torch.arange(0, max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2) * -(math.log(10000.0) / embed_size))
    encodings = torch.zeros(1, max_length, embed_size)
    encodings[0, :, 0::2] = torch.sin(position * div_term)
    encodings[0, :, 1::2] = torch.cos(position * div_term)
    return encodings

--- in the Encoder

        positions = positional_encoding(self.max_length, self.embed_size).to(self.device)

--- in the Decoder

        positions = positional_encoding(self.max_length, self.embed_size).to(self.device)

'''

# we just want to change this to include sinusoidal encoding
