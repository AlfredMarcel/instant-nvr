import torch.nn as nn
#import spconv
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
from typing import *
# from hash_embedder import HashEmbedder
from lib.networks.make_network import make_deformer, make_part_network, make_skinning_network
from termcolor import cprint
from lib.utils.render_utils import save_point_cloud
from lib.utils.base_utils import get_time

def gradient(input: torch.Tensor, output: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False):
    if d_out is not None:
        d_output = d_out
    elif isinstance(output, torch.Tensor):
        d_output = torch.ones_like(output, requires_grad=False)
    else:
        d_output = [torch.ones_like(o, requires_grad=False) for o in output]
    grads = torch.autograd.grad(inputs=input,
                                outputs=output,
                                grad_outputs=d_output,
                                create_graph=create_graph,
                                retain_graph=retain_graph,
                                only_inputs=True,)
    if len(grads) == 1:
        return grads[0]  # return the gradient directly
    else:
        return grads  # to be expanded


def compute_val_pair_around_range(pts: torch.Tensor, decoder: Callable[[torch.Tensor], torch.Tensor], diff_range: float, precomputed: torch.Tensor = None):
    # sample around input point and compute values
    # pts and its random neighbor are concatenated in second dimension
    # if needed, decoder should return multiple values together to save computation
    n_batch, n_pts, D = pts.shape
    # print_cyan(n_pts)
    neighbor = pts + (torch.rand_like(pts) - 0.5) * diff_range
    if precomputed is None:
        full_pts = torch.cat([pts, neighbor], dim=1)  # cat in n_masked dim
        raw: torch.Tensor = decoder(full_pts)  # (n_batch, n_masked, 3)
    else:
        nei = decoder(neighbor)  # (n_batch, n_masked, 3)
        raw = torch.cat([precomputed, nei], dim=1)
    return raw


class GradModule(nn.Module):
    # GradModule is a module that takes gradient based on whether we're in training mode or not
    # Avoiding the high memory cost of retaining graph of *not needed* back porpagation
    def __init__(self):
        super(GradModule, self).__init__()

    def gradient(self, input: torch.Tensor, output: torch.Tensor, d_out: torch.Tensor = None, create_graph: bool = False, retain_graph: bool = False) -> torch.Tensor:
        return gradient(input, output, d_out, self.training or create_graph, self.training or retain_graph)

    def jacobian(self, input: torch.Tensor, output: torch.Tensor):
        with torch.enable_grad():
            outputs = output.split(1, dim=-1)
        grads = [self.gradient(input, o, retain_graph=(i < len(outputs))) for i, o in enumerate(outputs)]
        jac = torch.stack(grads, dim=-1)
        return jac


class Network(GradModule):
    def __init__(self, init_network=True):
        super(Network, self).__init__()

        if not cfg.part_deform:
            self.tpose_deformer = make_deformer(cfg)
        self.tpose_human = TPoseHuman()
        self.mweight_vol_decoder = make_skinning_network(cfg)

        assert cfg.use_knn
    
    def sample_mweights(self, pose_pts, pose_dirs, batch):
        # 返回：1.变换后的采样点, 2. 变换后的采样方向 3.采样点在pose空间下的蒙皮权重(算skinning loss)
        
        # 初始化 skinning_network
        # [25,32,32,32]
        motion_weights_priors = batch['motion_weights_priors']
        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors)[0]

        pose_pts = pose_pts.reshape(-1,3)
        pose_dirs = pose_dirs.reshape(-1,3)
        # 暂不考虑skinning网络预测的 点到smpl距离
        motion_weights = motion_weights_vol[:-1]

        gltms = torch.inverse(batch['A']+1e-12)
        cnl_tms = torch.matmul(batch['big_A'], gltms)
        motion_scale_Rs = cnl_tms[:, :, :3, :3][0]
        motion_Ts = cnl_tms[:, :, :3, 3][0]

        weights_list = []
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i,:,:], pose_pts.T).T + motion_Ts[i,:]
            pos = (pos - batch['cnl_bbox_min_xyz'])* batch['cnl_bbox_scale_xyz'] - 1.0
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :,:,:],
                                    grid=pos[None,None,None,:,:].float(),
                                    padding_mode='zeros',
                                    align_corners=True )
            weights = weights[0,0,0,0,:,None]
            weights_list.append(weights)
        bw_weights = torch.cat(weights_list, dim=-1)
        total_bases = bw_weights.shape[-1]

        bw_weights_sum = torch.sum(bw_weights, dim=-1, keepdim=True)

        pts_motion_fields = []
        dirs_motion_fields = []
        for i in range(total_bases):
            pos = torch.matmul(motion_scale_Rs[i,:,:], pose_pts.T).T + motion_Ts[i,:]
            weighted_pos = bw_weights[:,i:i+1] * pos
            pts_motion_fields.append(weighted_pos)

            dir = torch.matmul(motion_scale_Rs[i,:,:], pose_dirs.T).T
            weighted_dir = bw_weights[:,i:i+1] * dir
            dirs_motion_fields.append(weighted_dir)
        
        x_skel = torch.sum(torch.stack(pts_motion_fields, dim=0), dim=0)/bw_weights_sum.clamp(min=0.0001)
        x_dir = torch.sum(torch.stack(dirs_motion_fields, dim=0), dim=0)/bw_weights_sum.clamp(min=0.0001)
        
        return x_skel[None,:,:], x_dir[None,:,:], bw_weights

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i
        # tmp = pose_pts.squeeze()[:64].cpu().numpy()
        # save_point_cloud(tmp, "debug/pose_pts_{}.ply".format(get_time()))
        B, N, _ = pose_pts.shape
        assert B == 1
        with torch.no_grad():
            assert batch['ppts'].shape[0] == 1
            # init_pbw = pts_knn_blend_weights_multiassign(pose_pts, batch['ppts'], batch['weights'], batch['parts'])  # (B, N, P, 24)

            # 采样点距离分块smpl表面的距离
            # init_pbw = pts_knn_blend_weights_multiassign_batch(pose_pts, batch['part_pts'][0], batch['part_pbw'][0], batch['lengths2'][0])  # (B, N, P, 24)
            # pred_pbw, pnorm = init_pbw[..., :24], init_pbw[..., 24]
            pnorm = pts_knn_dists_multiassign_batch(pose_pts, batch['part_pts'][0], None, batch['lengths2'][0])
            pflag = pnorm < cfg.smpl_thresh  # (B, N, P)

            # 采样点最近邻smpl顶点真值蒙皮权重
            pbw = pts_knn_blend_weights(pose_pts, batch['ppts'], batch['weights'])
            pbw = pbw[0][...,:-1]

        pflag = pflag.reshape(B, N*NUM_PARTS)

        # 使用 skinning_weights_network 预测的蒙皮权重
        wraped_pts, wraped_dir, mbw = self.sample_mweights(pose_pts,pose_dirs,batch)
        # 并在此处将预测权重mbw与 真值pbw 添加一个蒙皮权重 Loss  (ActorNeRF)
        resd_bw = mbw - pbw

        # transform points from i to i_0
        # A_bw, R_inv = get_inverse_blend_params(pred_pbw, batch['A'])
        # big_A_bw = get_blend_params(pred_pbw, batch['big_A'])

        # init_tpose = pose_points_to_tpose_points(pose_pts_part_extend, A_bw=A_bw, R_inv=R_inv)  # (B, N*P, 3)
        # init_bigpose = tpose_points_to_pose_points(init_tpose, A_bw=big_A_bw)  # (B, N*P, 3)

        # if cfg.tpose_viewdir and pose_dirs is not None:
        #     init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs_part_extend, A_bw=A_bw, R_inv=R_inv)
        #     tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, A_bw=big_A_bw).reshape(B, N, NUM_PARTS, 3)
        # else:
        #     tpose_dirs = None

        init_bigpose = wraped_pts[:,:,None,:].expand(B,N,NUM_PARTS,3).reshape(B,N*NUM_PARTS,3) 
        tpose_dirs = wraped_dir[:,:,None,:].expand(B,N,NUM_PARTS,3)

        assert cfg.part_deform == False

        resd = self.tpose_deformer(init_bigpose, batch, flag=pflag)
        tpose = init_bigpose + resd
        # tmp = tpose.squeeze()[:64].detach().cpu().numpy()

        tpose = tpose.reshape(B, N, NUM_PARTS, 3)
        bigpose = init_bigpose.reshape(B, N, NUM_PARTS, 3)
        pflag = pflag.reshape(B, N, NUM_PARTS)
        resd = resd.reshape(B, N, NUM_PARTS, 3)

        # print(type(tmp))
        # save_point_cloud(tmp, "debug/tpose_pts_{}.ply".format(get_time()))
        return tpose, bigpose, tpose_dirs, resd, pflag, init_bigpose, pnorm, resd_bw

    def resd(self, tpts: torch.Tensor, batch):
        B, N, D = tpts.shape
        return self.tpose_deformer(tpts, batch).view(B, N, D)  # has batch dimension

    def forward(self, wpts: torch.Tensor, viewdir: torch.Tensor, dists: torch.Tensor, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]  # B, N, 3, fake batch dimension
        viewdir = viewdir[None]  # B, N, 3

        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])
        # 筛去插值后的pbw最后一维<smpl_thresh的点
        with torch.no_grad():
            pnorm = pts_sample_blend_weights(pose_pts, batch['pbw'][..., -1:], batch['pbounds']) 
            pnorm = pnorm[0, -1]  # B, 25, N -> N,
            pind = pnorm < cfg.smpl_thresh  # N,
            pind = pind.nonzero(as_tuple=True)[0][:, None].expand(-1, 3)  # N, remove uncessary sync
            viewdir = viewdir[0].gather(dim=0, index=pind)[None]
            pose_pts = pose_pts[0].gather(dim=0, index=pind)[None]
            pose_dirs = pose_dirs[0].gather(dim=0, index=pind)[None]

        # transform points from the pose space to the tpose space
        tpose, bigpose, tpose_dirs, resd, tpose_part_flag, tpts, part_dist, resd_bw = self.pose_points_to_tpose_points(pose_pts, pose_dirs, batch)
        
        # 渲染canonical space：即不对光线wrap
        if cfg.render_canonical:
            assert self.eval
            B, N, _ = pose_pts.shape
            NUM_PARTS = 5
            tpose = pose_pts[:,:,None,:].expand(B,N,NUM_PARTS,3)
            bigpose = pose_pts[:,:,None,:].expand(B,N,NUM_PARTS,3)
            tpose_dirs = pose_dirs[:,:,None,:].expand(B,N,NUM_PARTS,3)

        tpose = tpose[0]
        bigpose = bigpose[0]
        tpose_part_flag = tpose_part_flag[0]
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]

        # part network query with hashtable embedding
        ret = self.tpose_human(tpose, bigpose, viewdir, tpose_part_flag, dists, part_dist, batch)

        B, N = wpts.shape[:2]
        assert B == 1
        raw = torch.zeros((B, N, 4+cfg.t_features_dim), device=wpts.device, dtype=wpts.dtype)
        occ = torch.zeros((B, N, 1), device=wpts.device, dtype=wpts.dtype)
        raw[0, pind[:, 0]] = ret['raw']  # NOTE: ignoring batch dimension
        occ[0, pind[:, 0]] = ret['occ']

        ret.update({'raw': raw, 'occ': occ, })

        if self.training:
            tocc = ret['tocc'].view(B, -1, 1)  # tpose occ, correpond with tpts (before deform)
            ret.update({'resd': resd, 'tpts': tpts, 'tocc': tocc, 'resd_bw': resd_bw})
        else:
            del ret['tocc']
        return ret


class TPoseHuman(GradModule):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        # self.head_bbox = torch.Tensor([ # 313
        #     [-0.3, 0.1, -0.3],
        #     [0.3, 0.7, 0.3]
        # ]).cuda()

        self.part_networks = nn.ModuleList([make_part_network(cfg, p, i) for i, p in enumerate(partnames)])
        if not cfg.silent:
            print("Finish initialize part networks")

    def save_part_decoders(self):
        for partnet in self.part_networks:
            partnet.save_decoder()
        self.body_network.save_decoder()

    def save_parts(self):
        for partnet in self.part_networks:
            partnet.save_part()
        self.body_network.save_part()

    def forward(self, tpts: torch.Tensor, bigpts: torch.Tensor, viewdir: torch.Tensor, tflag: torch.Tensor, dists: torch.Tensor, part_dist, batch):
        """
        """
        assert not cfg.part_deform

        # prepare inputs
        N = tpts.shape[0]
        raws = torch.zeros(N, NUM_PARTS, 4 + cfg.t_features_dim, device=tpts.device)
        occs = torch.zeros(N, NUM_PARTS, 1, device=tpts.device)

        # computing indices
        inds = []
        for part_idx in range(NUM_PARTS):
            flag_part = tflag[:, part_idx]  # flag_part: N, assuming viewdir and xyz have same dim
            flag_inds = flag_part.nonzero(as_tuple=True)[0]  # When input is on CUDA, torch.nonzero() causes host-device synchronization.
            inds.append(flag_inds)

        # applying mask
        xyz_parts = []
        rigid_parts = []
        viewdir_parts = []
        for part_idx in range(NUM_PARTS):
            xyz_part = tpts[:, part_idx].gather(dim=0, index=inds[part_idx][:, None].expand(-1, 3))  # faster backward than indexing, using resd so need backward
            rigid_part = bigpts[:, part_idx].gather(dim=0, index=inds[part_idx][:, None].expand(-1, 3))  # faster backward than indexing, using resd so need backward
            viewdir_part = viewdir[:, part_idx].gather(dim=0, index=inds[part_idx][:, None].expand(-1, 3))
            xyz_parts.append(xyz_part)
            rigid_parts.append(rigid_part)
            viewdir_parts.append(viewdir_part)

        # forward network
        ret_parts = []
        for part_idx in range(NUM_PARTS):
            xyz_part = xyz_parts[part_idx]
            xyzt_part = torch.cat((xyz_part, batch["frame_dim"].unsqueeze(0).expand(xyz_part.shape[0], -1)), dim=1)
            rigid_part = rigid_parts[part_idx]
            viewdir_part = viewdir_parts[part_idx]
            part_network = self.part_networks[part_idx]
            ret_part = part_network(xyz_part, xyzt_part, rigid_part, viewdir_part, dists, batch)
            ret_parts.append(ret_part)

        # fill in output
        for part_idx in range(NUM_PARTS):
            flag_inds = inds[part_idx]
            ret_part = ret_parts[part_idx]
            raws[flag_inds, part_idx] = ret_part['raw'].to(raws.dtype, non_blocking=True)
            occs[flag_inds, part_idx] = ret_part['occ'].to(occs.dtype, non_blocking=True)

        if cfg.aggr == 'mean':
            raw = raws.mean(dim=1)
            occ = occs.mean(dim=1)
            return {'raw': raw, 'occ': occ, 'tocc': occs}
        elif cfg.aggr == 'dist':
            part_dist_inv = F.normalize(1.0 / (part_dist[0] + 1e-5), dim=-1)
            raw = torch.sum(raws * part_dist_inv[..., None], dim=1)
            occ = torch.sum(occs * part_dist_inv[..., None], dim=1)
            return {'raw': raw, 'occ': occ, 'tocc': occs}
        elif cfg.aggr == 'mindist':
            breakpoint()
            ind = part_dist[0, :, :, None].argmin(dim=1).reshape(N, 1, 1).expand(N, 1, 4)
            raw = torch.gather(raws, 1, ind)[:, 0, :]
            ind = part_dist[0, :, :, None].argmin(dim=1).reshape(N, 1, 1).expand(N, 1, 1)
            occ = torch.gather(occs, 1, ind)[:, 0, :]
            return {'raw': raw, 'occ': occ, 'tocc': occs}

        ind = occs.argmax(dim=1).reshape(N, 1, 1).expand(N, 1, 4+cfg.t_features_dim)
        raw = torch.gather(raws, 1, ind)[:, 0, :]
        occ = occs.max(dim=1)[0]
        return {'raw': raw, 'occ': occ, 'tocc': occs}

    def get_occupancy(self, tpts, batch):
        raise NotImplementedError
