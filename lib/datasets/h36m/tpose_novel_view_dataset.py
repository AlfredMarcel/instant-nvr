import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
import pickle
from lib.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils
from plyfile import PlyData
from lib.utils import render_utils
from lib.utils.blend_utils import NUM_PARTS, part_bw_map, partnames
from lib.utils.body_utils import approx_gaussian_bone_volumes, get_canonical_global_tfms


SHOW_FRAME = 100


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        #==============================================================
        # 高斯近似的 T-pose 权重先验
        self.canonical_joints, self.canonical_bbox = self.load_canonical_joints(human)
        # 
        self.motion_weights_priors = approx_gaussian_bone_volumes(self.canonical_joints,self.canonical_bbox['min_xyz'],self.canonical_bbox['max_xyz'],
                                                                   grid_size=cfg.mweight_volume.volume_size).astype('float32')
        #==============================================================


        # 数据集动作序列长度
        self.dataset_length = {"my_377":617, "my_386":646, "my_387":654, "my_394":859, "my_393":658, "my_392":556}

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        test_view = [cfg.novel_view]
        view = cfg.training_view if split == 'train' else test_view
        self.num_cams = len(view)
        K, RT = render_utils.load_cam(ann_file)
        # center = np.array([0., 0., 5.])
        render_w2c = render_utils.gen_path(RT)

        i = cfg.begin_ith_frame
        self.ims = np.array([
            np.array(ims_data['ims'])[cfg.training_view]
            for ims_data in annots['ims'][i:i + cfg.num_train_frame *
                                          cfg.frame_interval]
        ])

        self.K = K[0]
        self.render_w2c = render_w2c
        # base_utils.write_K_pose_inf(self.K, self.render_w2c, img_root)

        self.Ks = np.array(K)[cfg.training_view].astype(np.float32)
        self.RT = np.array(RT)[cfg.training_view].astype(np.float32)
        self.Ds = np.array(self.cams['D'])[cfg.training_view].astype(
            np.float32)

        self.lbs_root = os.path.join(self.data_root, cfg.lbs)
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

        self.nrays = cfg.N_rand

        if cfg.use_knn:
            faces, weights, joints, parents, parts = self.load_smpl()
            self.meta_smpl = {'faces': faces, 'weights': weights, 'joints': joints, 'parents': parents, 'parts': parts}

        # 子弹时间帧
        self.bullet_frame = self.__len__()//2
        self.bullet_frame_length = cfg.num_render_views

    def load_smpl(self):
        smpl_meta_root = cfg.smpl_meta
        faces = np.load(os.path.join(smpl_meta_root, 'faces.npy')).astype(np.int64)
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy')).astype(np.float32)
        parents = np.load(os.path.join(smpl_meta_root, 'parents.npy')).astype(np.int64)
        weights = np.load(os.path.join(smpl_meta_root, 'weights.npy')).astype(np.float32)
        parts = np.zeros((6890,))
        weights_max = weights.argmax(axis=-1)
        for pid in range(NUM_PARTS):
            partname = partnames[pid]
            part_bwids = part_bw_map[partname]
            for bwid in part_bwids:
                parts[weights_max == bwid] = pid

        return faces, weights, joints, parents, parts

    def load_canonical_joints(self, human):
        tab = {"my_377":"canonical_joints_377.pkl","my_386":"canonical_joints_386.pkl","my_394":"canonical_joints_394.pkl","my_387":"canonical_joints_387.pkl","my_392":"canonical_joints_392.pkl","my_393":"canonical_joints_393.pkl"}
        path = os.path.join("canonical_joints/", tab[human])
        with open(path, 'rb') as f:
            cl_joint_data = pickle.load(f)
            canonical_joints = cl_joint_data['joints_{}'.format(human[-3:])]
            canonical_bbox = self.skeleton_to_bbox(canonical_joints)
        return canonical_joints, canonical_bbox

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_overlap
        min_xyz.astype(np.float32)
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_overlap
        max_xyz.astype(np.float32)

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def prepare_input(self, i):
        if self.human in ['CoreView_313', 'CoreView_315']:
            i = i + 1

        if 'olek' in self.human or 'vlad' in self.human:
            i = i + 1

        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)
        big_poses = np.zeros_like(poses).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        # big_poses[23] = np.deg2rad(-angle)
        # big_poses[26] = np.deg2rad(angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, joints, parents)
        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th, big_A

    def get_mask(self, i):
        ims = self.ims[i]
        msks = []

        for nv in range(len(ims)):
            im = ims[nv]

            msk_path = os.path.join(self.data_root, 'mask_cihp',
                                    im)[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, im.replace(
                    'images', 'mask'))[:-4] + '.png'
            if not os.path.exists(msk_path):
                msk_path = os.path.join(self.data_root, im.replace(
                    'images', 'mask'))[:-4] + '.jpg'
            msk_cihp = imageio.imread(msk_path)
            if len(msk_cihp.shape) == 3:
                msk_cihp = msk_cihp[..., 0]
            if 'deepcap' in self.data_root:
                msk_cihp = (msk_cihp > 125).astype(np.uint8)
            else:
                msk_cihp = (msk_cihp != 0).astype(np.uint8)

            msk = msk_cihp.astype(np.uint8)

            K = self.Ks[nv].copy()
            K[:2] = K[:2] / cfg.ratio
            msk = cv2.undistort(msk, K, self.Ds[nv])

            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk = cv2.dilate(msk.copy(), kernel)

            msks.append(msk)

        return msks

    def __getitem__(self, index):
        view_index = index
        # breakpoint()
        if cfg.render_frame == -1:
            latent_index = index
        else:
            latent_index = cfg.render_frame

        # frame_index = cfg.begin_ith_frame + latent_index * cfg.frame_interval
        # cam_index = 0

        # frame_index = cfg.begin_ith_frame + (1 - 1) * cfg.frame_interval
        # frame_index = cfg.begin_ith_frame + 0 * cfg.frame_interval

        # cam_index = index % len(self.render_w2c)
        # cam_index = latent_index

        # read v_shaped
        vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = if_nerf_dutils.get_bounds(tpose)
        tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        tbw = tbw.astype(np.float32)

        cam_index = 0

        if index < self.bullet_frame:
            frame_index = index 
        elif index < self.bullet_frame + self.bullet_frame_length:
            frame_index = self.bullet_frame
            cam_index = index - self.bullet_frame
        else:
            frame_index = index - self.bullet_frame_length

        wpts, ppts, A, pbw, Rh, Th, big_A = self.prepare_input(frame_index)

        pbounds = if_nerf_dutils.get_bounds(ppts)
        wbounds = if_nerf_dutils.get_bounds(wpts)

        # msks = self.get_mask(frame_index)

        # reduce the image resolution by ratio
        # 推理过程中，image内容不需要，只需要image的分辨率
        img_path = os.path.join(self.data_root, self.ims[0][0])
        img = imageio.imread(img_path)
        H, W = img.shape[:2]
        H, W = int(H * cfg.ratio), int(W * cfg.ratio)
        # msks = [
        #     cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        #     for msk in msks
        # ]
        # msks = np.array(msks)

        K = self.K
        RT = self.render_w2c[cam_index]
        R, T = RT[:3, :3], RT[:3, 3:]
        ray_o, ray_d, near, far, mask_at_box = if_nerf_dutils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)
        # ray_o, ray_d, near, far, center, scale, mask_at_box = render_utils.image_rays(
        #         RT, K, wbounds)
        occupancy = np.zeros((ray_o.shape[0], 1))
        tuv = np.load(os.path.join(self.data_root, 'bigpose_uv.npy'))
        frame_dim = np.array(latent_index / cfg.num_train_frame, dtype=np.float32)

        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            "occupancy": occupancy,
            'mask_at_box': mask_at_box,
            'tuv': tuv,
            'frame_dim': frame_dim
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': big_A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        latent_index = min(latent_index, cfg.num_train_frame - 1)
        meta = {
            'R': R,
            'Th': Th,
            'latent_index': latent_index,
            'frame_index': frame_index,
            'view_index': view_index
        }
        ret.update(meta)

        # meta = {'msks': msks, 'Ks': self.Ks, 'RT': self.RT, 'H': H, 'W': W}
        meta = {'Ks': self.Ks, 'RT': self.RT, 'H': H, 'W': W}
        ret.update(meta)

        if cfg.use_knn:
            ret.update({'ppts': ppts, 'wpts': wpts, 'tpts': tpose})
            if cfg.n_coarse_knn_ref != -1:
                ret['sub_ppts'] = ppts[np.random.permutation(cfg.n_coarse_knn_ref)]
            else:
                ret['sub_ppts'] = ppts

            ret.update(self.meta_smpl)

            N, D = self.meta_smpl['weights'].shape
            P = NUM_PARTS

            pose_verts = ppts
            pose_bw = self.meta_smpl['weights']

            pts_part = self.meta_smpl['parts']

            part_pts = np.zeros((P, N, 3), dtype=np.float32)
            part_pbw = np.zeros((P, N, D), dtype=np.float32)
            lengths2 = np.zeros(P, dtype=int)
            bounds = np.zeros((P, 2, 3), dtype=np.float32)
            for pid in range(P):
                part_flag = (pts_part == pid)
                lengths2[pid] = np.count_nonzero(part_flag)
                part_pts[pid, :lengths2[pid]] = pose_verts[part_flag]
                part_pbw[pid, :lengths2[pid]] = pose_bw[part_flag]

                bounds[pid, 0] = tpose[part_flag].min(axis=0) - cfg.bbox_overlap
                bounds[pid, 1] = tpose[part_flag].max(axis=0) + cfg.bbox_overlap

            max_length = lengths2.max()
            part_pts = part_pts[:, :max_length, :]
            part_pbw = part_pbw[:, :max_length, :]

            ret.update({
                'part_pts': part_pts,
                'part_pbw': part_pbw,
                'lengths2': lengths2,
                'bounds': bounds,
            })

        ret.update({'motion_weights_priors': self.motion_weights_priors.copy()})
        ret.update({'cnl_bbox_min_xyz': self.canonical_bbox['min_xyz'],
                    'cnl_bbox_max_xyz': self.canonical_bbox['max_xyz'],
                    'cnl_bbox_scale_xyz': (2.0 / (self.canonical_bbox['max_xyz'] - self.canonical_bbox['min_xyz'])).astype(np.float32),
                    })

        return ret

    def __len__(self):
        # return len(self.render_w2c)
        # return cfg.num_train_frame
        return len(self.render_w2c) + self.dataset_length[self.human]
        # return cfg.num_latent_code
