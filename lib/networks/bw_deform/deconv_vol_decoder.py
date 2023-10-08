import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.network_utils import ConvDecoder3D

# skinning network
# 输出canonical space 下的蒙皮权重体积表示

class Network(nn.Module):
    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(Network, self).__init__()

        self.total_bones = total_bones
        self.volume_size = volume_size
        
        self.const_embedding = nn.Parameter(
            torch.zeros(embedding_size), requires_grad=True 
        )

        self.decoder = ConvDecoder3D(
            embedding_size=embedding_size,
            volume_size=volume_size, 
            voxel_channels=total_bones+1)


    def forward(self,
                motion_weights_priors,
                **_):
        embedding = self.const_embedding[None, ...]
        # print(embedding.shape)
        decoded_weights =  F.softmax(self.decoder(embedding) + \
                                        torch.log(motion_weights_priors), 
                                     dim=1)
        
        return decoded_weights
