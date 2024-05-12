import torch
from torch import nn
from einops import rearrange
import numpy as np
from typing import List
from models.id_embedding.helpers import get_rep_pos, shift_tensor_dim0
from models.id_embedding.meta_net import StyleVectorizer
from models.celeb_embeddings import _get_celeb_embeddings_basis

from functools import partial
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init


DEFAULT_PLACEHOLDER_TOKEN = ["*"]

PROGRESSIVE_SCALE = 2000

def get_clip_token_for_string(tokenizer, string):
    batch_encoding = tokenizer(string, return_length=True, padding=True, truncation=True, return_overflowing_tokens=False, return_tensors="pt")
    tokens = batch_encoding["input_ids"]
    
    return tokens 


def get_embedding_for_clip_token(embedder, token):
    return embedder(token.unsqueeze(0))


class EmbeddingManagerId_adain(nn.Module):
    def __init__(
            self,
            tokenizer,
            text_encoder,
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),  
            experiment_name = "normal_GAN",                      
            num_embeds_per_token: int = 2,  
            loss_type: str = None,
            mlp_depth: int = 2,    
            token_dim: int = 1024,   
            input_dim: int = 1024, 
            **kwargs
    ):
        super().__init__()
        self.device = device
        self.num_es = num_embeds_per_token

        self.get_token_for_string = partial(get_clip_token_for_string, tokenizer)        
        self.get_embedding_for_tkn = partial(get_embedding_for_clip_token, text_encoder.text_model.embeddings)  
        

        self.token_dim = token_dim

        ''' 1. Placeholder mapping dicts '''
        self.placeholder_token = self.get_token_for_string("*")[0][1]    
        
        if experiment_name == "normal_GAN":
            self.celeb_embeddings_mean, self.celeb_embeddings_std = _get_celeb_embeddings_basis(tokenizer, text_encoder, "datasets_face/good_names.txt")
        elif experiment_name == "man_GAN":
            self.celeb_embeddings_mean, self.celeb_embeddings_std = _get_celeb_embeddings_basis(tokenizer, text_encoder, "datasets_face/good_names_man.txt")
        elif experiment_name == "woman_GAN":            
            self.celeb_embeddings_mean, self.celeb_embeddings_std = _get_celeb_embeddings_basis(tokenizer, text_encoder, "datasets_face/good_names_woman.txt")
        else:
            print("Hello, please notice this ^_^")
            assert 0
        print("now experiment_name:", experiment_name)
        
        self.celeb_embeddings_mean = self.celeb_embeddings_mean.to(device)   
        self.celeb_embeddings_std = self.celeb_embeddings_std.to(device)  

        self.name_projection_layer = StyleVectorizer(input_dim, self.token_dim * self.num_es, depth=mlp_depth, lr_mul=0.1) 
        self.embedding_discriminator = Embedding_discriminator(self.token_dim * self.num_es, dropout_rate = 0.2)

        self.adain_mode = 0
        
    def forward(
            self,
            tokenized_text, 
            embedded_text, 
            name_batch,
            random_embeddings = None,
            timesteps = None,
    ):
        
        if tokenized_text is not None:
            batch_size, n, device = *tokenized_text.shape, tokenized_text.device
        other_return_dict = {}
        
        if random_embeddings is not None:
            mlp_output_embedding = self.name_projection_layer(random_embeddings)   
            total_embedding = mlp_output_embedding.view(mlp_output_embedding.shape[0], 2, 1024)   

            if self.adain_mode == 0:          
                adained_total_embedding = total_embedding * self.celeb_embeddings_std + self.celeb_embeddings_mean
            else:
                adained_total_embedding = total_embedding
                
            other_return_dict["total_embedding"] = total_embedding
            other_return_dict["adained_total_embedding"] = adained_total_embedding

        if name_batch is not None:
            if isinstance(name_batch, list): 
                name_tokens = self.get_token_for_string(name_batch)[:, 1:3]
                name_embeddings = self.get_embedding_for_tkn(name_tokens.to(random_embeddings.device))[0] 
                
                other_return_dict["name_embeddings"] = name_embeddings
            else:
                assert 0

        if tokenized_text is not None:
            placeholder_pos = get_rep_pos(tokenized_text,
                                        [self.placeholder_token])
            placeholder_pos = np.array(placeholder_pos)
            if len(placeholder_pos) != 0:
                batch_size = adained_total_embedding.shape[0]  
                end_index = min(batch_size, placeholder_pos.shape[0]) 
                embedded_text[placeholder_pos[:, 0], placeholder_pos[:, 1]] = adained_total_embedding[:end_index,0,:]
                embedded_text[placeholder_pos[:, 0], placeholder_pos[:, 1] + 1] = adained_total_embedding[:end_index,1,:]

        return embedded_text, other_return_dict



    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda')
        if ckpt.get("name_projection_layer") is not None:
            self.name_projection_layer = ckpt.get("name_projection_layer").float()

        print('[Embedding Manager] weights loaded.')



    def save(self, ckpt_path):
        save_dict = {}
        save_dict["name_projection_layer"] = self.name_projection_layer
        
        torch.save(save_dict, ckpt_path)


    def trainable_projection_parameters(self):  
        trainable_list = []
        trainable_list.extend(list(self.name_projection_layer.parameters())) 

        return trainable_list
   


class Embedding_discriminator(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super(Embedding_discriminator, self).__init__()
        self.input_size = input_size

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)

        self.LayerNorm1 = nn.LayerNorm(512)
        self.LayerNorm2 = nn.LayerNorm(256)

        self.leaky_relu = nn.LeakyReLU(0.2)

        self.dropout_rate = dropout_rate
        if self.dropout_rate > 0:
            self.dropout1 = nn.Dropout(dropout_rate)
            self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, input):
        x = input.view(-1, self.input_size)

        if self.dropout_rate > 0:
            x = self.leaky_relu(self.dropout1(self.fc1(x)))
        else:
            x = self.leaky_relu(self.fc1(x))
        
        if self.dropout_rate > 0:
            x = self.leaky_relu(self.dropout2(self.fc2(x)))
        else:
            x = self.leaky_relu(self.fc2(x))

        x = self.fc3(x)

        return x
    
    
    def save(self, ckpt_path):
        save_dict = {}
 
        save_dict["fc1"] = self.fc1
        save_dict["fc2"] = self.fc2
        save_dict["fc3"] = self.fc3
        save_dict["LayerNorm1"] = self.LayerNorm1
        save_dict["LayerNorm2"] = self.LayerNorm2
        save_dict["leaky_relu"] = self.leaky_relu
        save_dict["dropout1"] = self.dropout1
        save_dict["dropout2"] = self.dropout2
        
        torch.save(save_dict, ckpt_path)
    
    def load(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cuda')
        
        if ckpt.get("first_name_proj_layer") is not None:
            self.fc1 = ckpt.get("fc1").float()
            self.fc2 = ckpt.get("fc2").float()
            self.fc3 = ckpt.get("fc3").float()
            self.LayerNorm1 = ckpt.get("LayerNorm1").float()
            self.LayerNorm2 = ckpt.get("LayerNorm2").float()
            self.leaky_relu = ckpt.get("leaky_relu").float()
            self.dropout1 = ckpt.get("dropout1").float()
            self.dropout2 = ckpt.get("dropout2").float()
            
        print('[Embedding D] weights loaded.')



    