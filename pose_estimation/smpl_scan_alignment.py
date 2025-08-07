from scipy.spatial import cKDTree
import numpy as np
import torch
import open3d as o3d
import pathlib

def read_from_pickle(path):
    objects = []
    with open(path, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    return objects

def transfer_smplx_pose_to_smpl(smplx_pose_pkl):
    smpl_pose = np.zeros((23, 3), dtype=float)
    smpl_pose[:21] = smplx_pose_pkl['body_pose'].reshape(21, 3)
    smpl_pose[21] = smplx_pose_pkl['left_hand_pose'].reshape(15, 3)[0]
    smpl_pose[22] = smplx_pose_pkl['right_hand_pose'].reshape(15, 3)[0]
    return smpl_pose

def init_SMPL_layer(gender, initial_params, device):
    params = {}
    if initial_params is None:
        params["pose_params"] = torch.zeros(1, 69, device=device)
        params["shape_params"] = torch.zeros(1, 10, device=device)
        params["scale"] = torch.ones(1, device=device)
    else:
        params["pose_params"] = torch.tensor(
            initial_params["pose_params"], dtype=torch.float32, device=device
        )
        params["shape_params"] = torch.tensor(
            initial_params["shape_params"], dtype=torch.float32, device=device
        )

    params["trans"] = torch.zeros(1, 3, device=device)
    params["pose_params"].requires_grad = True
    params["shape_params"].requires_grad = True
    params["scale"].requires_grad = True
    params["trans"].requires_grad = True

    model_folder = r"C:\Users\natalie.n.griffin4\Documents\DA\models"
    smpl_layer = smplx.create(
        model_path=model_folder,
        model_type="smpl",
        gender=gender,
        betas=params["shape_params"],
        body_pose=params["pose_params"],
    ).to(device)

    return smpl_layer, params

def optimize_SMPL_to_scan(
    scan_pts, gender="neutral", initial_params=None, num_iters=100, device="cuda"
):
    smpl_layer, params = init_SMPL_layer(gender, initial_params, device)

    target = torch.tensor(scan_pts, dtype=torch.float32, device=device)
    scan_floor = scan_pts[:, 1].min()
    
    with torch.no_grad():
        init_verts = smpl_layer(
            body_pose=params["pose_params"], betas=params["shape_params"]
        ).vertices

    smpl_floor = init_verts[:, 1].min()
    initial_trans = np.array([0, scan_floor - smpl_floor, 0], dtype=np.float32)
    params["trans"].data = torch.tensor(initial_trans, device=device).unsqueeze(0)

    optimizer = torch.optim.Adam(
        [
            {"params": params["pose_params"]},
            {"params": params["shape_params"]},
            {"params": params["scale"]},
            {"params": params["trans"]},
        ],
        lr=1e-3,
    )

    for it in range(num_iters):
        optimizer.zero_grad()
        # Loss computation and optimization steps would go here

    final_model = smpl_layer(
        body_pose=params["pose_params"], betas=params["shape_params"]
    )

    return (
        params["pose_params"].detach().cpu().numpy(),
        params["shape_params"].detach().cpu().numpy(),
        params["scale"].item(),
        params["trans"].detach().cpu().numpy(),
        final_model,
    )

if __name__ == "__main__":
    scan_file = r"C:\Users\natalie.n.griffin4\Documents\DA\squat\squat_meters.ply"
    scan_mesh = o3d.io.read_triangle_mesh(scan_file)
    scan_pts = np.asarray(scan_mesh.vertices)

    result_pose = read_from_pickle(
        r"C:\Users\natalie.n.griffin4\Documents\DA\squat\pose3d_smpl_fused\fused_joints_LS_alpha3d75_ik.pkl"
    )[0]
    init_params = {
        "pose_params": transfer_smplx_pose_to_smpl(result_pose).reshape(1, 69),
        "shape_params": result_pose["betas"],
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pose, shape, scale, trans, model = optimize_SMPL_to_scan(
        scan_pts, gender="neutral", initial_params=init_params, num_iters=150, device=device
    )

    verts = model.vertices.detach().cpu().numpy().squeeze()
    verts = verts * scale + trans

    smpl_pc = o3d.geometry.PointCloud()
    smpl_pc.points = o3d.utility.Vector3dVector(verts.astype(np.float32))
    smpl_pc.paint_uniform_color([0.6, 0.3, 0.7])

    output_file = r"C:\Users\natalie.n.griffin4\Documents\DA\squat\smpl_pc.ply"
    o3d.io.write_point_cloud(output_file, smpl_pc)

    scan_pc = o3d.geometry.PointCloud()
    scan_pc.points = o3d.utility.Vector3dVector(scan_pts.astype(np.float32))
    scan_pc.paint_uniform_color([0.2, 0.5, 0.3])

    o3d.visualization.draw_geometries([smpl_pc, scan_pc])
```