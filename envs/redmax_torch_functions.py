from typing import Any, Tuple

import torch
import torch.autograd as autograd

import redmax_py as redmax
import numpy as np
import time
from utils.torch_utils import *

class EpisodicSimFunction(autograd.Function):

    @staticmethod
    def forward(
            ctx: Any, 
            q0: torch.Tensor, 
            qdot0: torch.Tensor,
            actions: torch.Tensor, 
            tactile_masks: torch.Tensor,
            sim: redmax.Simulation,
            grad_mode: bool
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        T = actions.shape[0]

        ctx.sim = sim
        ctx.T = T
        ctx.grad_q0 = q0.requires_grad
        ctx.grad_qdot0 = qdot0.requires_grad
        ctx.grad_actions = actions.requires_grad
        ctx.device = q0.device
        ctx.dtype = q0.dtype
        ctx.tactile_masks = tactile_masks

        q0_np = q0.detach().cpu().numpy()
        qdot0_np = qdot0.detach().cpu().numpy()
        actions_np = actions.detach().cpu().numpy()

        sim.set_state_init(q0_np, qdot0_np)

        sim.reset(backward_flag = grad_mode)

        qs = []
        vars = []
        tactiles = []
        
        for t in range(T):
            sim.set_u(actions_np[t])
            sim.forward(1, verbose = False, test_derivatives = False)

            q = sim.get_q().copy()
            qs.append(to_torch(q, dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode))
            var = sim.get_variables().copy()
            vars.append(to_torch(var, dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode))

            if tactile_masks[t]:
                tactile = sim.get_tactile_force_vector().copy()
                tactiles.append(to_torch(tactile, dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode))
            
        qs = torch.stack(qs, dim = 0)
        vars = torch.stack(vars, dim = 0)
        tactiles = torch.stack(tactiles, dim = 0)

        if grad_mode:
            sim.saveBackwardCache()

        return qs, vars, tactiles

    # TODO: change c++ for tactile masks
    @staticmethod
    def backward(
            ctx: Any,
            df_dq: torch.Tensor,
            df_dvar: torch.Tensor,
            df_dtactile: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None, None, None]:
        
        sim = ctx.sim
        T = ctx.T

        sim.popBackwardCache()

        sim.backward_info.set_flags(flag_q0 = ctx.grad_q0, flag_qdot0 = ctx.grad_qdot0, flag_p = False, flag_u = ctx.grad_actions)

        sim.backward_info.df_dq = df_dq.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dvar = df_dvar.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dtactile = df_dtactile.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dq0 = np.zeros(sim.ndof_r)
        sim.backward_info.df_dqdot0 = np.zeros(sim.ndof_r)
        sim.backward_info.df_du = np.zeros(sim.ndof_u * T)

        sim.backward()

        if ctx.grad_q0:
            df_dq0 = to_torch(sim.backward_results.df_dq0.copy(), dtype = ctx.dtype, device = ctx.device, requires_grad = True)
        else:
            df_dq0 = None

        if ctx.grad_qdot0:
            df_dqdot0 = to_torch(sim.backward_results.df_dqdot0.copy(), dtype = ctx.dtype, device = ctx.device, requires_grad = True)
        else:
            df_dqdot0 = None

        if ctx.grad_actions:
            df_du = to_torch(sim.backward_results.df_du.copy().reshape(T, sim.ndof_u), dtype = ctx.dtype, device = ctx.device, requires_grad = True)
        else:
            df_du = None
        
        return df_dq0, df_dqdot0, df_du, None, None, None


class StepSimFunction(autograd.Function):

    @staticmethod
    def forward(
            ctx: Any, 
            action: torch.Tensor, 
            num_steps: int, 
            sim: redmax.Simulation,
            grad_mode: bool
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        ctx.sim = sim
        ctx.num_steps = num_steps
        ctx.grad_actions = action.requires_grad
        ctx.device = action.device
        ctx.dtype = action.dtype

        action_np = action.detach().cpu().numpy()
        
        sim.set_u(action_np)
        sim.forward(num_steps, verbose = False, test_derivatives = False, save_last_frame_var_only = True)

        q = to_torch(sim.get_q().copy(), dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode)
        var = to_torch(sim.get_variables().copy(), dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode)
        tactile = to_torch(sim.get_tactile_force_vector(), dtype = ctx.dtype, device = ctx.device, requires_grad = grad_mode)

        return q, var, tactile

    @staticmethod
    def backward(
            ctx: Any,
            df_dq: torch.Tensor,
            df_dvar: torch.Tensor,
            df_dtactile: torch.Tensor
        ) -> Tuple[torch.Tensor, None, None, None]:
        
        sim = ctx.sim
        num_steps = ctx.num_steps

        sim.backward_info.set_flags(flag_q0 = False, flag_qdot0 = False, flag_p = False, flag_u = ctx.grad_actions)

        df_dq_full = np.zeros(sim.ndof_r * num_steps)
        df_dq_full[-sim.ndof_r:] = df_dq.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dq = df_dq_full

        df_dvar_full = np.zeros(sim.ndof_var * num_steps)
        df_dvar_full[-sim.ndof_var:] = df_dvar.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dvar = df_dvar_full

        df_dtactile_full = np.zeros(sim.ndof_tactile * num_steps)
        df_dtactile_full[-sim.ndof_tactile:] = df_dtactile.view(-1).detach().cpu().numpy()
        sim.backward_info.df_dtactile = df_dtactile_full

        sim.backward_info.df_du = np.zeros(sim.ndof_u * num_steps)

        sim.backward_steps(num_steps)

        if ctx.grad_actions:
            df_du = to_torch(sim.backward_results.df_du.copy().reshape(num_steps, sim.ndof_u), dtype = ctx.dtype, device = ctx.device, requires_grad = True)
        else:
            df_du = None
        
        return df_du, None, None, None



