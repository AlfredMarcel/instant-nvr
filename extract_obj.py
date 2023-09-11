import torch
import numpy as np

from pytorch3d.io import save_obj


def smpl2obj(verts,file_path):

    faces = np.load('data/smpl-meta/faces.npy').astype(np.float32)

    if verts.shape[0]==1:
        verts = torch.squeeze(verts,dim=0)

    v = torch.tensor(verts,dtype=torch.float32)
    fa = torch.tensor(faces,dtype=torch.float32)
    save_obj(
        f = file_path,
        verts = v,
        faces = fa,
    )

data_path = 'data/zju-mocap/my_377/smpl_vertices/214.npy'
verts = np.load(data_path)

smpl2obj(verts,'377_214.obj')
