"""
Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

import torch
import numpy as np

def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False):
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)

# reference: https://i.imgur.com/Mdc2AV3.jpg
def rotvec_mul(a, b):
    a_norm = a.norm()
    b_norm = b.norm()
    
    if a_norm < 1e-7:
        return b
    if b_norm < 1e-7:
        return a
    
    a_unit = a / a_norm
    b_unit = b / b_norm
    
    c_norm = 2. * torch.arccos(torch.cos(a_norm / 2.) * torch.cos(b_norm / 2.) - torch.dot(a_unit * torch.sin(a_norm / 2.), b_unit * torch.sin(b_norm / 2.)))

    if c_norm < 1e-7:
        return torch.zeros(3)

    c_unit = (torch.cos(a_norm / 2.) * torch.sin(b_norm / 2.) * b_unit + torch.cos(b_norm / 2.) * torch.sin(a_norm / 2.) * a_unit + torch.cross(a_unit * torch.sin(a_norm / 2.), b_unit * torch.sin(b_norm / 2.))) / torch.sin(c_norm / 2.)

    return c_norm * c_unit

@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def normalize(x, eps: float = 1e-9):
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


@torch.jit.script
def quat_apply(a, b):
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, :3]
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 3:] * t + xyz.cross(t, dim=-1)).view(shape)


@torch.jit.script
def quat_rotate(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c


@torch.jit.script
def quat_rotate_inverse(q, v):
    shape = q.shape
    q_w = q[:, -1]
    q_vec = q[:, :3]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * \
        torch.bmm(q_vec.view(shape[0], 1, 3), v.view(
            shape[0], 3, 1)).squeeze(-1) * 2.0
    return a - b + c


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


@torch.jit.script
def quat_unit(a):
    return normalize(a)


@torch.jit.script
def quat_from_angle_axis(angle, axis):
    theta = (angle / 2).unsqueeze(-1)
    xyz = normalize(axis) * theta.sin()
    w = theta.cos()
    return quat_unit(torch.cat([xyz, w], dim=-1))

@torch.jit.script
def quat_from_rotvec(rotvec):
    angle = torch.linalg.norm(rotvec, dim = -1)
    axis = rotvec / torch.clip(angle, min = 1e-6).unsqueeze(-1)
    return quat_from_angle_axis(angle, axis)

@torch.jit.script
def normalize_angle(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


@torch.jit.script
def tf_inverse(q, t):
    q_inv = quat_conjugate(q)
    return q_inv, -quat_apply(q_inv, t)


@torch.jit.script
def tf_apply(q, t, v):
    return quat_apply(q, v) + t


@torch.jit.script
def tf_vector(q, v):
    return quat_apply(q, v)


@torch.jit.script
def tf_combine(q1, t1, q2, t2):
    return quat_mul(q1, q2), quat_apply(q1, t2) + t1


@torch.jit.script
def get_basis_vector(q, v):
    return quat_rotate(q, v)


def get_axis_params(value, axis_idx, x_value=0., dtype=np.float32, n_dims=3):
    """construct arguments to `Vec` according to axis index.
    """
    zs = np.zeros((n_dims,))
    assert axis_idx < n_dims, "the axis dim should be within the vector dimensions"
    zs[axis_idx] = 1.
    params = np.where(zs == 1., value, zs)
    params[0] = x_value
    return list(params.astype(dtype))


@torch.jit.script
def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float32).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)


@torch.jit.script
def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


@torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)


@torch.jit.script
def torch_rand_float(lower, upper, shape, device):
    # type: (float, float, Tuple[int, int], str) -> Tensor
    return (upper - lower) * torch.rand(*shape, device=device) + lower


@torch.jit.script
def torch_random_dir_2(shape, device):
    # type: (Tuple[int, int], str) -> Tensor
    angle = torch_rand_float(-np.pi, np.pi, shape, device).squeeze(-1)
    return torch.stack([torch.cos(angle), torch.sin(angle)], dim=-1)


@torch.jit.script
def tensor_clamp(t, min_t, max_t):
    return torch.max(torch.min(t, max_t), min_t)


@torch.jit.script
def scale(x, lower, upper):
    return (0.5 * (x + 1.0) * (upper - lower) + lower)


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


def unscale_np(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)

def remap(x, lower_prev, upper_prev, lower, upper):
    return (x - lower_prev) / (upper_prev - lower_prev) * (upper - lower) + lower

# @torch.jit.script
def random_quaternions(num, device, dtype = torch.float32, order = 'xyzw'):
    ran = torch.rand(num, 3, dtype=dtype, device=device)
    r1, r2, r3 = ran[:, 0], ran[:, 1], ran[:, 2]
    pi2 = 2 * np.pi
    r1_1 = torch.sqrt(1.0 - r1)
    r1_2 = torch.sqrt(r1)
    t1 = pi2 * r2
    t2 = pi2 * r3

    quats = torch.zeros(num, 4, dtype=dtype, device=device)

    if order == 'wxyz':
        quats[:, 0] = r1_1 * (torch.sin(t1))
        quats[:, 1] = r1_1 * (torch.cos(t1))
        quats[:, 2] = r1_2 * (torch.sin(t2))
        quats[:, 3] = r1_2 * (torch.cos(t2))
    else:
        quats[:, 3] = r1_1 * (torch.sin(t1))
        quats[:, 0] = r1_1 * (torch.cos(t1))
        quats[:, 1] = r1_2 * (torch.sin(t2))
        quats[:, 2] = r1_2 * (torch.cos(t2))

    return quats

def grad_norm(params):
    grad_norm = 0.
    for p in params:
        if p.grad is not None:
            grad_norm += torch.sum(p.grad ** 2)
    return torch.sqrt(grad_norm)

def flatten_grad(network):
    dofs = 0
    for params in network.parameters():
        dofs += torch.numel(params.data)
    grad_vec = torch.zeros(dofs)
    offset = 0
    for params in network.parameters():
        if params.grad is not None:
            grad_vec[offset:offset + torch.numel(params.data)] = params.grad.clone().view(-1)
        offset += torch.numel(params.data)
    return grad_vec

def flatten_params(network):
    dofs = 0
    for params in network.parameters():
        dofs += torch.numel(params.data)
    params_vec = torch.zeros(dofs)
    offset = 0
    for params in network.parameters():
        params_vec[offset:offset + torch.numel(params.data)] = params.data.clone().view(-1)
        offset += torch.numel(params.data)
    return params_vec

def fill_params(network, params_vec):
    offset = 0
    for params in network.parameters():
        params.data = params_vec[offset:offset + torch.numel(params.data)].view(params.data.shape)
        offset += torch.numel(params.data)