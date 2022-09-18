import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import grad
try:
    from pytorch3d.loss.mesh_laplacian_smoothing import cot_laplacian
except:
    from pytorch3d.loss.mesh_laplacian_smoothing import laplacian_cot as cot_laplacian


def make_grid(h, w, device, dtype):
    grid_x = torch.linspace(-1, 1, w, device=device, dtype=dtype)
    grid_y = torch.linspace(-1, 1, h, device=device, dtype=dtype)
    v, u = torch.meshgrid(grid_y, grid_x)
    grid = torch.stack([u, v], dim=2).view(1, h * w, 2)

    return grid


class Transform(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(Transform, self).__init__()
        self.sigma_affine = sigma_affine
        self.sigma_tps = sigma_tps
        self.points_tps = points_tps

    def transform_img(self, img):
        b, _, h, w = img.shape
        device = img.device
        dtype = img.dtype

        if not hasattr(self, 'identity_grid'):
            identity_grid = make_grid(h, w, device, dtype)
            self.register_buffer('identity_grid', identity_grid, persistent=False)

        if not hasattr(self, 'control_grid'):
            control_grid = make_grid(self.points_tps, self.points_tps, device, dtype)
            self.register_buffer('control_grid', control_grid, persistent=False)

        # Sample transform
        noise = torch.normal(
            mean=0,
            std=self.sigma_affine,
            size=(b, 2, 3),
            device=device,
            dtype=dtype)

        self.theta = (noise + torch.eye(2, 3, device=device, dtype=dtype)[None])[:, None]  # b x 1 x 2 x 3

        self.control_params = torch.normal(
            mean=0,
            std=self.sigma_tps,
            size=(b, 1, self.points_tps ** 2),
            device=device,
            dtype=dtype)

        grid = self.warp_pts(self.identity_grid).view(-1, h, w, 2)

        return F.grid_sample(img, grid, padding_mode="reflection")

    def warp_pts(self, pts):
        b = self.theta.shape[0]
        n = pts.shape[1]
 
        pts_transformed = torch.matmul(self.theta[:, :, :, :2], pts[..., None]) + self.theta[:, :, :, 2:]
        pts_transformed = pts_transformed[..., 0]

        pdists = pts[:, :, None] - self.control_grid[:, None]
        pdists = (pdists).abs().sum(dim=3)

        result = pdists**2 * torch.log(pdists + 1e-5) * self.control_params
        result = result.sum(dim=2).view(b, n, 1)

        pts_transformed = pts_transformed + result

        return pts_transformed

    def jacobian(self, pts):
        new_pts = self.warp_pts(pts)
        grad_x = grad(new_pts[..., 0].sum(), pts, create_graph=True)
        grad_y = grad(new_pts[..., 1].sum(), pts, create_graph=True)
        jac = torch.cat([grad_x[0].unsqueeze(-2), grad_y[0].unsqueeze(-2)], dim=-2)
        
        return jac


class EquivarianceLoss(nn.Module):
    def __init__(self, sigma_affine, sigma_tps, points_tps):
        super(EquivarianceLoss, self).__init__()
        self.transform = Transform(sigma_affine, sigma_tps, points_tps)

    def forward(self, img, kp, jac, kp_detector):
        img_transformed = self.transform.transform_img(img)
        kp_transformed, jac_transformed = kp_detector(img_transformed)
        kp_recon = self.transform.warp_pts(kp_transformed)

        loss_kp = (kp - kp_recon).abs().mean()

        jac_recon = torch.matmul(self.transform.jacobian(kp_transformed), jac_transformed)
        inv_jac = torch.linalg.inv(jac)

        loss_jac = (torch.matmul(inv_jac, jac_recon) - torch.eye(2)[None, None].type(inv_jac.type())).abs().mean()

        return loss_kp, loss_jac, img_transformed, kp_transformed, kp_recon


class LaplaceMeshLoss(nn.Module):
    def __init__(self, type='uniform', use_vector_constant=False):
        super(LaplaceMeshLoss, self).__init__()
        self.method = type
        self.precomputed_laplacian = None
        self.use_vector_constant = use_vector_constant

    def _compute_loss(self, L, verts_packed, inv_areas=None):
        if self.method == "uniform":
            loss = L.mm(verts_packed)
        
        elif self.method == "cot":
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            idx = norm_w > 0
            norm_w[idx] = 1.0 / norm_w[idx]
            loss = L.mm(verts_packed) * norm_w - verts_packed
        
        elif self.method == "cotcurv":
            L_sum = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            norm_w = 0.25 * inv_areas
            loss = (L.mm(verts_packed) - L_sum * verts_packed) * norm_w
        
        return loss.norm(dim=1)

    def forward(self, meshes, coefs=None):
        if meshes.isempty():
            return torch.tensor(
                [0.0], dtype=torch.float32, device=meshes.device, requires_grad=True
            )

        N = len(meshes)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        faces_packed = meshes.faces_packed()  # (sum(F_n), 3)
        num_verts_per_mesh = meshes.num_verts_per_mesh()  # (N,)
        verts_packed_idx = meshes.verts_packed_to_mesh_idx()  # (sum(V_n),)
        weights = num_verts_per_mesh.gather(0, verts_packed_idx)  # (sum(V_n),)
        weights = 1.0 / weights.float()
        norm_w, inv_areas = None, None
        with torch.no_grad():
            if self.method == "uniform":
                if self.precomputed_laplacian is None or self.precomputed_laplacian.shape[0] != verts_packed.shape[0]:
                    L = meshes.laplacian_packed()
                    self.precomputed_laplacian = L
                else:
                    L = self.precomputed_laplacian
            elif self.method in ["cot", "cotcurv"]:
                L, inv_areas = cot_laplacian(verts_packed, faces_packed)
            else:
                raise ValueError("Method should be one of {uniform, cot, cotcurv}")

        loss = self._compute_loss(L, verts_packed,
                                  inv_areas=inv_areas)
        loss = loss * weights
        if coefs is not None:
            loss = loss * coefs.view(-1)

        return loss.sum() / N