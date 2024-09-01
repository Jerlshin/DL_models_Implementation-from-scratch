"""
- Vision Transformer 
- Contrastive Learning (CLIP, SigLip)
- Laguage Model (Gemma)
- KV-Cache
- Rotary Position Embedding 
- Normalization (Batch, Layer, RMS)
"""

# softmax transforms LLM generated values to distribution which sums up to 1 
# softmax = exp(x) / sum(exp(x))  -- exp() is a function that grows very fast, basically by its name 
# if exponential of the exp() is too big, we would use up lot of memory 
# So, make """Numerical Stability of Softmax """ -- means the numbers should be represented withing hte memory block like 16 bit or 32 bit. 


from typing import Dict, List, Optional, Union, Tuple, Iterable
import numpy as np 
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5] 
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]

"""Image tokens + Text tokens"""
# prepends a sequence of placeholder image tokens "<image>" to a text prompt. it is used to ensure that the image embeddings are aligned with the text embeddings in the final input sequence
def add_image_tokens_to_prompt(
    prefix_prompt,  # the text prompt to which the image tokens will be added
    bos_token,  # bos token
    image_seq_len,  # number of image tokens to prepend  
    image_token  # the token representing an image in the model's tokenizer
):
    # a new string that begine with the image tokens, followed by bos token and then original prompt 
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"

""" img...... ,bos, prompt......, sep, output""" # sep - seperator to distinguish between the input and the output.. sep = "\n" # nextline character     
# we tokenize SEP token seperately to avoid it being merged by the tokenizer with either the end of the prefix or the begining of the suffix 
# this function resizes an image to the specified size using a chosen resampling
def resize(
    image: Image,  # input
    size: Tuple[int, int],   # of the output  
    resample: Image.Resampling = None,   # Bicubic 
    reducing_gap: Optional[int] = None,   # to control how much of the image should be preprocessed before resizing. 
) -> np.ndarray:
    
    # Resampling algorithms that interpolate pixel values to create the resized image. 
    """
    # Nearest Neighbor, 
    # Bilinear Interpolation,
    # Bicubic Interpolation,
    # Lanczos Resampling
    """
      
    height, width = size 
    resized_image = image.resize(  
        (width, height), resample=resample, reducing_gap=reducing_gap
    ) 
    
    return resized_image

def normalize(
    image: np.ndarray, 
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    
    image = (image - mean) / std # (x - myu) / std
    
    return image 
    

# function to rescale the pixel values of an image to the range [0, 1] by multiplying with 1/255. 
def rescale(
    image: np.ndarray,  # many arrays as for each image in the list of images 
    scale: float,  # the factor by which to multiply the image's pixel values 
    dtype: np.dtype = np.float32 # to rescaled the image to this
) -> np.ndarray:
    
    rescaled_image = image * scale # broadcasting 
    rescaled_image = rescaled_image.astype(dtype)  # 
    
    return rescaled_image
    
    
def process_images(
    images: List[Image.Image],  # list of PIL Image
    size: Dict[str, int] = None,   # specifying the target size for image resizing. - "height" : __, "width": ___ 
    # Image.Resampling is an enumeration provided by the PIL. it defines different resampling algorithms used when resizing images. 
    resample: Image.Resampling = None, # to use when resizing the image 
    rescale_factor: float = None,  # used to scale pixel values to range  of [0, 1]
    # Union indicates that a value can be one of several specified types. 
    image_mean: Optional[Union[float, List[float]]] = None,  # either be float or list of floats. Optional means that it can even be None
    image_std: Optional[Union[float, List[float]]] = None,
) -> List[np.ndarray]:  # returns a list of Numpy arrays, where each array represents a processed image 
    
    # usually this would be keys like "height" and "width"
    height, width = size[0], size[1]  # got from the Dict[str, int]
    
    # resizes each image in the list to the specified dim using the method 
    images = [
        # resamplling: process of resizing an image and adjusting its pixel values  ---   adjust the dim of an image
        resize(image=image, size=(height, width), resample=resample) for image in images 
    ]
    # for resizing use PIL and then use Numpy
    # convert each image to a numpy array 
    images = [np.array(image) for image in images] # converts PIL image to Numpy array. easier to manipulate after resizing.
    
    # rescale the pixel value to be in range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]     
    
    # Normalize the images to have mean 0 and standard deviation 1 
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]
    
    # Move the channel dimesntion to the first dim. 
    images = [images.transpose(2, 0, 1) for image in images]
    
    return images 

class PaliGemmaProcessor:
    # placeholder tokens for the image embeddings. 
    # we first process the text with the Gemma tokenizer and then we concatenate the image embeddings with the text embeddings. 
    # for this, while processing the text, we need to have placeholder for the image embeddings, so, for that, we use <image> tokens
    IMAGE_TOKEN = "<image>"
    def __init__(
        self, 
        tokenizer, 
        num_image_tokens: int,   # how many image tokens needed to be generated for the iamge 
        image_size: int, # size of the input image size 
    ):
        super().__init__()
         
        self.image_seq_length = num_image_tokens
        self.image_size = image_size
        
        # gemma tokenizers are used for the text but there is no specific tokenizer for the image for the gemma model 
        # Tokenizer. add the image token to the tokenizer. 
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]} # dictionary to add special tokens to the tokenizer 
        tokenizer.add_special_tokens(tokens_to_add)  # add_special_tokens
        
        """Paligemma can do multiple task, so, it can do localization and segmentation and detection... extended tokens  """
        # tokens for the localization
        EXTRA_TOKENS = [  # for the bouding bozm regions, we need this 1024 tokens 
            f"<loc>" for i in range(1024) # tokens are used for object detection (bounding boxes)
        ]
        
        # tokens for the segementation
        EXTRA_TOKENS += [
            f"<seg" for i in range(128)  # for object segmentation 
        ]
        
        # add all the tokens    
        tokenizer.add_tokens(EXTRA_TOKENS)
        # convert the tokens to ids suing the tokenizer 
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN) # 
        
        # BOS - beginning of sequence TOKEN
        tokenizer.add_bos_token = False
        # EOS - end of sequence TOKEN
        tokenizer.add_eos_token = False
        
        # stores the tokenizer instance 
        self.tokenizer = tokenizer
    
    # it will work with only 1 image and 1 prompt at a time. 
    # can be used as a function 
    def __call__(
        self, 
        # we accept 1 image and 1 text as per this code as we don't need to complicate this code by padding the stuff
        text: List[str], 
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict: 
        
        # ensure that exactly one image and one prompt is given 
        assert len(images) == 1 and len(text), f"Received {len(images)} images for {len(text)} prompts."
        
        # process the image 
        pixel_values = process_images(
            images, 
            size=(self.image_size, self.image_size), 
            resample=Image.Resampling.BICUBIC,   # bicubic resampling 
            rescale_factor=1/255.0,   # to be within a range of [0, 1]
            image_mean=IMAGENET_STANDARD_MEAN, # normalizing the distribution
            image_std=IMAGENET_STANDARD_STD,
        )
        
        # convert the list of numpy arrays to a single numpy array with shape [batch, channel, height, width]
        # stack all the List of processed images from the function along batch dimension. 
        pixel_values = np.stack(pixel_values, axis=0) # across first dim (rows). so batches 
        # convert the numpy array to a pytorch tensor 
        pixel_values = torch.tensor(pixel_values)

        # prepend a 'self.image_seq_length' number of image tokens to the prompt ----  so, that image embeddi8ngs are added to the text embeddings later
        input_strings = [
            add_image_tokens_to_prompt(  # crucial for multimodal models that use image embeddings combined with text embeddings. 
                prefix_prompt=prompt,                   # prompt 
                bos_token=self.tokenizer.bos_token,     # bos 
                image_seq_len=self.image_seq_length,    # place holder for image tokens
                image_token=self.IMAGE_TOKEN,           
            )  # also the new line character 
            for prompt in text  # we have only one prompt, so, this would go once. But, if we have multiple images and texts, we need to do other techniques like padding and other stuff 
        ]
        
        # input_ids -- ids from vocab corresponding to each of the word. 
        # returns the input_ids and attention_mask as tensors 
        inputs = self.tokenizer(  # modified text prompt is tokenized
            input_strings,  
            return_tensors="pt",
            padding=padding,
            truncation=truncation,
        )
        
        return_data = {"pixel_values": pixel_values, **inputs}  # ** is used to unpack the key-value paries from the dictionary and include them directly 
        
        return return_data
    
        
        