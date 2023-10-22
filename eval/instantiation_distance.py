import torch, numpy as np
import os, os.path as osp
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm


@torch.no_grad()
def compute_instantiation_distance_pair(
    A, B, device=torch.device("cuda:0"), N_states_max=20, N_pcl_max=2048, chunk=1000
):
    # just compute the dist between a pair, not batched
    x1_list, pose1_list = A  # N,PCL,3; N,P,4,4
    x2_list, pose2_list = B  #

    P1, P2 = pose1_list.shape[1], pose2_list.shape[1]
    N_states, N_pcl = x1_list.shape[0], x1_list.shape[1]
    assert N_states == x2_list.shape[0] and N_pcl == x2_list.shape[1]
    N_states = min(N_states, N_states_max)
    N_pcl = min(N_pcl, N_pcl_max)
    x1_list = x1_list[:N_states, :N_pcl, :]
    x2_list = x2_list[:N_states, :N_pcl, :]
    pose1_list = pose1_list[:N_states]
    pose2_list = pose2_list[:N_states]

    x1_list, pose1_list = x1_list.float().to(device), pose1_list.float().to(device)
    x2_list, pose2_list = x2_list.float().to(device), pose2_list.float().to(device)

    # canonicalize to all possible poses
    x1_list = torch.cat([x1_list, torch.ones_like(x1_list[..., :1])], dim=-1)  # N_states, PCL,4
    x2_list = torch.cat([x2_list, torch.ones_like(x2_list[..., :1])], dim=-1)
    inv_pose1_list = torch.inverse(pose1_list)  # N_states, P,4,4
    inv_pose2_list = torch.inverse(pose2_list)

    x1_list = torch.einsum("npij,ntj->npti", inv_pose1_list, x1_list)[..., :3]
    x2_list = torch.einsum("npij,ntj->npti", inv_pose2_list, x2_list)[..., :3]

    # compute N_states x N_states distance matrix
    # each row is a A, each col is a B
    D = []
    cur = 0
    while cur < N_states:
        # prepare all computing pairs
        src = x1_list[cur : cur + chunk]  # chunk,P1,PCL,3
        dst = x2_list.to(device)  # N_states,P2,PCL,3
        src = src[:, None, :, None, ...].expand(-1, N_states, -1, P2, -1, -1)
        dst = dst[None, :, None, ...].expand(len(src), -1, P1, -1, -1, -1)
        cd, _ = chamfer_distance(
            src.reshape(-1, N_pcl, 3), dst.reshape(-1, N_pcl, 3), batch_reduction=None
        )
        cd = cd.reshape(len(src), N_states, -1).min(dim=-1).values
        # Try all canonicalization and find the best one
        D.append(cd)
        cur += chunk
    D = torch.cat(D, dim=0)  # N_states, N_states

    dl, dr = D.min(dim=1).values, D.min(dim=0).values
    dl, dr = dl.mean(), dr.mean()
    distance = dl + dr  # ! note, it's sum

    # gather them in correct way
    return float(distance.cpu().numpy())


def compute_D_matrix(gen_dir, ref_dir, save_dir, N_states_max=10, N_pcl_max=2048):
    gen_fn_list = [f for f in os.listdir(gen_dir) if f.endswith(".npz")]
    ref_fn_list = [f for f in os.listdir(ref_dir) if f.endswith(".npz")]
    gen_fn_list.sort()
    ref_fn_list.sort()
    N_gen, N_ref = len(gen_fn_list), len(ref_fn_list)
    D = -1.0 * np.ones((N_gen, N_ref), dtype=np.float32)
    gen_name = osp.basename(gen_dir)
    ref_name = osp.basename(ref_dir)
    save_name = f"{gen_name}_{ref_name}_{N_states_max}_{N_pcl_max}.npz"
    save_fn = osp.join(save_dir, save_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"save to {save_fn}")
    # D: N_gen, N_ref

    sym_flag = gen_dir == ref_dir

    # cache DATA
    DATA_GEN, DATA_REF = [], []
    print("caching GEN ...")
    for i in tqdm(range(N_gen)):
        fn = osp.join(gen_dir, gen_fn_list[i])
        data = np.load(fn)
        pcl, pose = torch.from_numpy(data["pcl"]), torch.from_numpy(data["pose"])
        DATA_GEN.append((pcl, pose))
    if sym_flag:
        DATA_REF = DATA_GEN
    else:
        print("caching REF ...")
        for i in tqdm(range(N_ref)):
            fn = osp.join(ref_dir, ref_fn_list[i])
            data = np.load(fn)
            pcl, pose = torch.from_numpy(data["pcl"]), torch.from_numpy(data["pose"])
            DATA_REF.append((pcl, pose))

    for i in tqdm(range(N_gen)):
        for j in tqdm(range(N_ref)):
            # fn1 = osp.join(gen_dir, gen_fn_list[i])
            # fn2 = osp.join(ref_dir, ref_fn_list[j])
            # data1 = np.load(fn1)
            # data2 = np.loa(fn2)
            # pcl1, pose1 = torch.from_numpy(data1["pcl"]), torch.from_numpy(data1["pose"])
            # pcl2, pose2 = torch.from_numpy(data2["pcl"]), torch.from_numpy(data2["pose"])
            if sym_flag and i == j:
                D[i, j] = 0.0
                continue
            if sym_flag and i > j:
                assert D[j, i] >= 0.0
                D[i, j] = D[j, i]
                continue
            pcl1, pose1 = DATA_GEN[i]
            pcl2, pose2 = DATA_REF[j]
            _d = compute_instantiation_distance_pair(
                (pcl1, pose1),
                (pcl2, pose2),
                N_states_max=N_states_max,
                N_pcl_max=N_pcl_max,
                chunk=1000,
            )
            D[i, j] = _d
    assert (D >= 0.0).all(), "invalid D"
    np.savez_compressed(
        save_fn,
        D=D,
        gen_dir=gen_dir,
        ref_dir=ref_dir,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
    )
    np.savez_compressed(
        save_fn,
        D=D,
        gen_dir=gen_dir,
        ref_dir=ref_dir,
        N_states_max=N_states_max,
        N_pcl_max=N_pcl_max,
        gen_fn_list=gen_fn_list,
        ref_fn_list=ref_fn_list,
    )
    return D


if __name__ == "__main__":
    import time
    import argparse

    # TODO: can boost when compute self-self distance

    # prepare args for src, dst
    arg_parser = argparse.ArgumentParser(description="Run")
    arg_parser.add_argument(
        "--gen",
        # default="../log/test/K_8_cate_all_gt_samples",
        default="../log/test/PCL/K_8_cate_all_v5.1.5_5455/",
    )
    arg_parser.add_argument(
        "--ref",
        default="../log/test/PCL/K_8_cate_all_gt",
    )
    arg_parser.add_argument(
        "--save_dir",
        default="../log/test/ID_D_matrix",
    )
    arg_parser.add_argument("--n_states", default=10, type=int)
    arg_parser.add_argument("--n_pcl", default=2048, type=int)
    args = arg_parser.parse_args()

    # fn1 = "../log/test/K_8_cate_all_gt_samples/100753.npz"
    # fn2 = "../log/test/K_8_cate_all_gt_samples/103593.npz"

    # data1 = np.load(fn1)
    # data2 = np.load(fn2)

    # pcl1, pose1 = torch.from_numpy(data1["pcl"]), torch.from_numpy(data1["pose"])
    # pcl2, pose2 = torch.from_numpy(data2["pcl"]), torch.from_numpy(data2["pose"])

    # start_t = time.time()
    # for _ in tqdm(range(450**2)):
    #     compute_instantiation_distance_pair(
    #         (pcl1, pose1), (pcl2, pose2), N_states_max=10, N_pcl_max=2048, chunk=1000
    #     )
    # print((time.time() - start_t) / 1000.0)

    compute_D_matrix(gen_dir=args.gen, ref_dir=args.ref, save_dir=args.save_dir, N_states_max=args.n_states, N_pcl_max=args.n_pcl)
