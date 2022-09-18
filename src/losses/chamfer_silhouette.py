import torch
import torch.nn.functional as F
from torch import nn
from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

from typing import Union



class ChamferSilhouetteLoss(nn.Module):
    def __init__(
        self, 
        num_neighbours=1, 
        use_same_number_of_points=False, 
        sample_outside_of_silhouette=False,
        use_visibility=True
    ):
        super(ChamferSilhouetteLoss, self).__init__()
        self.num_neighbours = num_neighbours
        self.use_same_number_of_points = use_same_number_of_points
        self.sample_outside_of_silhouette = sample_outside_of_silhouette
        self.use_visibility = use_visibility

    def forward(self, 
                pred_points: torch.Tensor,
                points_visibility: torch.Tensor,
                target_silhouette: torch.Tensor,
                target_segs: torch.Tensor) -> torch.Tensor:        
        target_points, target_lengths, weight = self.get_pointcloud(target_segs, target_silhouette)

        if self.use_visibility:
            pred_points, pred_lengths = self.get_visible_points(pred_points, points_visibility)
                
        if self.use_same_number_of_points:
            target_points = target_points[:, :pred_points.shape[1]]    

            target_lengths = pred_lengths = torch.minimum(target_lengths, pred_lengths)
            
            if self.sample_outside_of_silhouette:
                target_lengths = (target_lengths.clone() * weight).long()

            for i in range(target_points.shape[0]):
                target_points[i, target_lengths[i]:] = -100.0

            for i in range(pred_points.shape[0]):
                pred_points[i, pred_lengths[i]:] = -100.0

        visible_batch = target_lengths > 0
        if self.use_visibility:
            visible_batch *= pred_lengths > 0

        if self.use_visibility:
            loss = chamfer_distance(
                pred_points[visible_batch], 
                target_points[visible_batch], 
                x_lengths=pred_lengths[visible_batch], 
                y_lengths=target_lengths[visible_batch],
                num_neighbours=self.num_neighbours
            )        
        else:
            loss = chamfer_distance(
                pred_points[visible_batch], 
                target_points[visible_batch], 
                y_lengths=target_lengths[visible_batch],
                num_neighbours=self.num_neighbours
            )

        if isinstance(loss, tuple):
            loss = loss[0]
        
        return loss, pred_points, target_points
    
    @torch.no_grad()
    def get_pointcloud(self, seg, silhouette):
        if self.sample_outside_of_silhouette:
            silhouette = (silhouette > 0.0).type(seg.type())

            old_area = seg.view(seg.shape[0], -1).sum(1)
            seg = seg * (1 - silhouette)
            new_area = seg.view(seg.shape[0], -1).sum(1)

            weight = new_area / (old_area + 1e-7)
        
        else:
            weight = torch.ones(seg.shape[0], dtype=seg.dtype, device=seg.device)

        batch, coords = torch.nonzero(seg[:, 0] > 0.5).split([1, 2], dim=1)
        batch = batch[:, 0]
        coords = coords.float()
        coords[:, 0] = (coords[:, 0] / seg.shape[2] - 0.5) * 2
        coords[:, 1] = (coords[:, 1] / seg.shape[3] - 0.5) * 2

        pointcloud = -100.0 * torch.ones(seg.shape[0], seg.shape[2]*seg.shape[3], 2).to(seg.device)
        length = torch.zeros(seg.shape[0]).to(seg.device).long()
        for i in range(seg.shape[0]):
            pt = coords[batch == i]
            pt = pt[torch.randperm(pt.shape[0])] # randomly permute the points
            pointcloud[i][:pt.shape[0]] = torch.cat([pt[:, 1:], pt[:, :1]], dim=1)
            length[i] = pt.shape[0]
        
        return pointcloud, length, weight
    
    @staticmethod
    def get_visible_points(points, visibility):
        batch, indices = torch.nonzero(visibility > 0.0).split([1, 1], dim=1)
        batch = batch[:, 0]
        indices = indices[:, 0]

        length = torch.zeros(points.shape[0]).to(points.device).long()
        for i in range(points.shape[0]):
            batch_i = batch == i
            indices_i = indices[batch_i]
            points[i][:indices_i.shape[0]] = points[i][indices_i]
            points[i][indices_i.shape[0]:] = -100.0
            length[i] = indices_i.shape[0]

        return points, length


# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def _validate_chamfer_reduction_inputs(
    batch_reduction: Union[str, None], point_reduction: str
):
    """Check the requested reductions are valid.
    Args:
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    """
    if batch_reduction is not None and batch_reduction not in ["mean", "sum"]:
        raise ValueError('batch_reduction must be one of ["mean", "sum"] or None')
    if point_reduction not in ["mean", "sum"]:
        raise ValueError('point_reduction must be one of ["mean", "sum"]')


def _handle_pointcloud_input(
    points: Union[torch.Tensor, Pointclouds],
    lengths: Union[torch.Tensor, None],
    normals: Union[torch.Tensor, None],
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if isinstance(points, Pointclouds):
        X = points.points_padded()
        lengths = points.num_points_per_cloud()
        normals = points.normals_padded()  # either a tensor or None
    elif torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError("Expected points to be of shape (N, P, D)")
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError("Expected lengths to be of shape (N,)")
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
        if normals is not None and normals.ndim != 3:
            raise ValueError("Expected normals to be of shape (N, P, 3")
    else:
        raise ValueError(
            "The input pointclouds should be either "
            + "Pointclouds objects or torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths, normals


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    x_normals=None,
    y_normals=None,
    weights=None,
    num_neighbours=1,
    batch_reduction: Union[str, None] = "mean",
    point_reduction: str = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.
    Args:
        x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
            a batch of point clouds with at most P1 points in each batch element,
            batch size N and feature dimension D.
        y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
            a batch of point clouds with at most P2 points in each batch element,
            batch size N and feature dimension D.
        x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in x.
        y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
            cloud in y.
        x_normals: Optional FloatTensor of shape (N, P1, D).
        y_normals: Optional FloatTensor of shape (N, P2, D).
        weights: Optional FloatTensor of shape (N,) giving weights for
            batch elements for reduction operation.
        batch_reduction: Reduction operation to apply for the loss across the
            batch, can be one of ["mean", "sum"] or None.
        point_reduction: Reduction operation to apply for the loss across the
            points, can be one of ["mean", "sum"].
    Returns:
        2-element tuple containing
        - **loss**: Tensor giving the reduced distance between the pointclouds
          in x and the pointclouds in y.
        - **loss_normals**: Tensor giving the reduced cosine distance of normals
          between pointclouds in x and pointclouds in y. Returns None if
          x_normals and y_normals are None.
    """
    _validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

    x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
    y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

    return_normals = x_normals is not None and y_normals is not None

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError("y does not have the correct shape.")
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError("weights must be of shape (N,).")
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    cham_norm_x = x.new_zeros(())
    cham_norm_y = x.new_zeros(())

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=num_neighbours)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=num_neighbours)

    cham_x = x_nn.dists.mean(-1)  # (N, P1)
    cham_y = y_nn.dists.mean(-1)  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    if return_normals:
        # Gather the normals using the indices and keep only value for k=0
        x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]
        y_normals_near = knn_gather(x_normals, y_nn.idx, x_lengths)[..., 0, :]

        cham_norm_x = 1 - torch.abs(
            F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
        )
        cham_norm_y = 1 - torch.abs(
            F.cosine_similarity(y_normals, y_normals_near, dim=2, eps=1e-6)
        )

        if is_x_heterogeneous:
            cham_norm_x[x_mask] = 0.0
        if is_y_heterogeneous:
            cham_norm_y[y_mask] = 0.0

        if weights is not None:
            cham_norm_x *= weights.view(N, 1)
            cham_norm_y *= weights.view(N, 1)

    # Apply point reduction
    cham_x = cham_x.sum(1)  # (N,)
    cham_y = cham_y.sum(1)  # (N,)
    if return_normals:
        cham_norm_x = cham_norm_x.sum(1)  # (N,)
        cham_norm_y = cham_norm_y.sum(1)  # (N,)
    if point_reduction == "mean":
        cham_x /= x_lengths
        cham_y /= y_lengths
        if return_normals:
            cham_norm_x /= x_lengths
            cham_norm_y /= y_lengths

    if batch_reduction is not None:
        # batch_reduction == "sum"
        cham_x = cham_x.sum()
        cham_y = cham_y.sum()
        if return_normals:
            cham_norm_x = cham_norm_x.sum()
            cham_norm_y = cham_norm_y.sum()
        if batch_reduction == "mean":
            div = weights.sum() if weights is not None else N
            cham_x /= div
            cham_y /= div
            if return_normals:
                cham_norm_x /= div
                cham_norm_y /= div

    cham_dist = cham_x + cham_y
    cham_normals = cham_norm_x + cham_norm_y if return_normals else None

    return cham_dist, cham_normals