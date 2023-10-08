import os
import sys
import torch
from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from smpl_numpy import SMPL
# from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags


MODEL_DIR = 'data/smpl-meta'

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

def main(argv):
    frames = 658
    smpl_params_dir = "data/zju-mocap/my_393/smpl_params"
    output_path = "data/zju-mocap/my_393"

    all_betas = []
    for i in range(frames):
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{i}.npy"),
            allow_pickle=True).item()
        betas = smpl_params['shapes'][0]
        all_betas.append(betas)

    # write mean joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL("neutral", model_dir=MODEL_DIR)

    big_poses = np.zeros(72)
    big_poses = big_poses.reshape(-1, 3)
    big_poses[1] = np.array([0, 0, 30. / 180. * np.pi])
    big_poses[2] = np.array([0, 0, -30. / 180. * np.pi])
    # big_poses[16] = np.array([0, 0, -55. / 180. * np.pi])
    # big_poses[17] = np.array([0, 0, 55. / 180. * np.pi])

    verts, template_joints = smpl_model(big_poses.reshape(-1), avg_betas)
    
    with open('mean_joints_393.pkl', 'wb') as f:   
        pickle.dump(
            {
                'joints_393': template_joints,
            }, f)
    
    smpl2obj(verts,"393.obj")

if __name__ == '__main__':
    app.run(main)
