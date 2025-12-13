#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#
import torch
import numpy as np

class Permutation(torch.nn.Module):

    def __init__(self, seq_length: int):
        super().__init__()
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        raise NotImplementedError('Overload me')


class PermutationIdentity(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x


class PermutationFlip(Permutation):
    def forward(self, x: torch.Tensor, dim: int = 1, inverse: bool = False) -> torch.Tensor:
        return x.flip(dims=[dim])


class Attention(torch.nn.Module):
    USE_SPDA: bool = True

    def __init__(self, in_channels: int, head_channels: int):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.GSJmode = 'J'

        self.k_cache: dict = {'cond': [], 'uncond': [], 'cond_temp': torch.tensor([]),'uncond_temp': torch.tensor([])}
        self.v_cache: dict = {'cond': [], 'uncond': [], 'cond_temp': torch.tensor([]),'uncond_temp': torch.tensor([])}

    def forward_spda(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).transpose(1, 2).chunk(3, dim=1)  # (b, h, t, d)

        if self.GSJmode == 'GSJ':
            self.k_cache[which_cache+'_temp'] = k
            self.v_cache[which_cache+'_temp'] = v
            if len(self.k_cache[which_cache]) == 0:
                k = self.k_cache[which_cache+'_temp']
                v = self.v_cache[which_cache+'_temp']
            else:
                k = torch.cat([torch.cat(self.k_cache[which_cache],dim=2), self.k_cache[which_cache+'_temp']],dim=2)
                v = torch.cat([torch.cat(self.v_cache[which_cache],dim=2), self.v_cache[which_cache+'_temp']],dim=2)
        
        elif self.GSJmode == 'GS':
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=scale)
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).reshape(B, T, 3 * self.num_heads, -1).chunk(3, dim=2)
        
        if self.GSJmode == 'GSJ':
            self.k_cache[which_cache+'_temp'] = k
            self.v_cache[which_cache+'_temp'] = v
            if len(self.k_cache[which_cache]) == 0:
                k = self.k_cache[which_cache+'_temp']
                v = self.v_cache[which_cache+'_temp']
            else:
                k = torch.cat([torch.cat(self.k_cache[which_cache],dim=2), self.k_cache[which_cache+'_temp']],dim=1)
                v = torch.cat([torch.cat(self.v_cache[which_cache],dim=2), self.v_cache[which_cache+'_temp']],dim=1)
        
        elif self.GSJmode == 'GS':
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=1)
            v = torch.cat(self.v_cache[which_cache], dim=1)

        attn = torch.einsum('bmhd,bnhd->bmnh', q * self.sqrt_scale, k * self.sqrt_scale) / temp
        if mask is not None:
            attn = attn.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
        attn = attn.float().softmax(dim=-2).type(attn.dtype)
        x = torch.einsum('bmnh,bnhd->bmhd', attn, v)
        x = x.reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        if self.USE_SPDA:
            return self.forward_spda(x, mask, temp, which_cache)
        return self.forward_base(x, mask, temp, which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(self, channels: int, head_channels: int, expansion: int = 4):
        super().__init__()
        self.attention = Attention(channels, head_channels)
        self.mlp = MLP(channels, expansion)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, attn_temp: float = 1.0, which_cache: str = 'cond'
    ) -> torch.Tensor:
        x = x + self.attention(x, attn_mask, attn_temp, which_cache)
        x = x + self.mlp(x)
        return x


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: Permutation,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        num_classes: int = 0,
        detect_mode: bool = False,
        norm: int = 2,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        if num_classes:
            self.class_embed = torch.nn.Parameter(torch.randn(num_classes, 1, channels) * 1e-2)
        else:
            self.class_embed = None
        self.attn_blocks = torch.nn.ModuleList(
            [AttentionBlock(channels, head_dim, expansion) for _ in range(num_layers)]
        )
        output_dim = in_channels * 2
        self.proj_out = torch.nn.Linear(channels, output_dim)
        self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer('attn_mask', torch.tril(torch.ones(num_patches, num_patches)))
        self.detect_mode = detect_mode
        self.norm = norm

    def forward(self, x: torch.Tensor, y: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.permutation(x)
        pos_embed = self.permutation(self.pos_embed, dim=0)
        x_in = x
        x = self.proj_in(x) + pos_embed
        if self.class_embed is not None:
            if y is not None:
                if (y < 0).any():
                    m = (y < 0).float().view(-1, 1, 1)
                    class_embed = (1 - m) * self.class_embed[y] + m * self.class_embed.mean(dim=0)
                else:
                    class_embed = self.class_embed[y]
                x = x + class_embed
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, self.attn_mask)
        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)

        xa, xc = x.chunk(2, dim=-1)
        alpha = (-xa.float()).exp().type(xa.dtype)
        gamma = xc
        z = alpha * (x_in - gamma)

        if self.detect_mode:
            norm = self.norm
            z0 = torch.cat([x_in[:,:1],torch.zeros_like(x_in[:, 1:])], dim=1)
            IGN = []
            for xinit in [z,z0]: 
                xinit = self.proj_in(xinit) + pos_embed
                if self.class_embed is not None:
                    xinit = xinit + self.class_embed.mean(dim=0)
                for block in self.attn_blocks:
                    xinit = block(xinit, self.attn_mask)
                xinit = self.proj_out(xinit)
                xinit = torch.cat([torch.zeros_like(xinit[:, :1]), xinit[:, :-1]], dim=1)
                inita, initc = xinit.chunk(2, dim=-1)
                initalpha = (inita.float()).exp().type(inita.dtype)
                initgamma = initc
                xadj = initalpha * z + initgamma
                res = xadj - x_in
                singular_value = torch.linalg.norm(res.mean(dim=0),ord=norm)
                IGN.append(singular_value.item())
            
            CRN = []
            singularsx = torch.linalg.norm((alpha * x_in).mean(dim=0),ord=norm)
            CRN.append(singularsx.item())

            W = self.proj_out.weight  
            Ws, Wu = W.chunk(2, dim=0)  
            singulars = torch.linalg.norm(Ws,ord=norm) 
            singularu = torch.linalg.norm(Wu,ord=norm)
            CRN.extend([singulars.item(), singularu.item()]) 
            CRN.append(CRN[0]*CRN[1]+CRN[2])

            return self.permutation(z , inverse=True), -xa.mean(dim=[1, 2]), np.round(np.array(IGN),2), np.round(np.array(CRN),2)
        else:
            return self.permutation(z , inverse=True), -xa.mean(dim=[1, 2])

    # function to calculate one sub_block of GS_Jacobi iteration    
    def reverse_substep(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        jacobi_attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = 'cond',
    ) -> tuple[torch.Tensor, torch.Tensor]:

        if self.class_embed is not None:
            if y is not None:
                x = x + self.class_embed[y]
            else:
                x = x + self.class_embed.mean(dim=0)

        for block in self.attn_blocks:
            x = block(x, jacobi_attn_mask, attn_temp=attn_temp, which_cache=which_cache)  
        x = self.proj_out(x)
        xa, xc = x.chunk(2, dim=-1)

        return xa, xc

    def set_GSJmode(self, mode: str = 'GSJ'):
        for m in self.modules():
            if isinstance(m, Attention):
                m.GSJmode = mode
                m.k_cache = {'cond': [], 'uncond': [], 'cond_temp': torch.tensor([]), 'uncond_temp': torch.tensor([])}
                m.v_cache = {'cond': [], 'uncond': [], 'cond_temp': torch.tensor([]), 'uncond_temp': torch.tensor([])}

    def cat_kv_temp(self,which_cache: str = 'cond'):
        for m in self.modules():
            if isinstance(m, Attention):
                m.k_cache[which_cache].append(m.k_cache[which_cache+'_temp'])
                m.v_cache[which_cache].append(m.v_cache[which_cache+'_temp'])

    def reverse(
        self,
        z: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ac',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        num_GS: int = 1,
        max_jacobi: int = 100,
        zero_guess: int = 0,
        ebound: float = 1e-8,
        show_trace: bool = False,
        X_target: torch.Tensor | None = None,

        incre1: bool = False,

    ) -> torch.Tensor:
        z = self.permutation(z)
        B, T, C = z.size()
        pos_embed = self.permutation(self.pos_embed, dim=0)
        
        if num_GS < 1 or num_GS == T:
            mode = 'GS'
        elif num_GS == 1:
            mode = 'J'
        elif num_GS > 1:
            mode = 'GSJ'
        self.set_GSJmode(mode)

        if num_GS < 1:
            num_GS = T
            max_jacobi = 1
        jacobi_size = T // num_GS

        if zero_guess == 0:
            x_next = z.clone()
        else:
            x_next = torch.cat([z[:,:1],torch.zeros_like(z[:, 1:])], dim=1)

        if mode == 'GS':
            for i in range(T-1):
                x_in = x_next[:, i : i + 1]
                x = self.proj_in(x_in) + pos_embed[i : i + 1]
                xa, xc = self.reverse_substep(x, y, jacobi_attn_mask=None,which_cache='cond')
                if guidance > 0 and guide_what:
                    xa_u, xc_u = self.reverse_substep(x, None, jacobi_attn_mask=None, attn_temp=attn_temp, which_cache='uncond')
                    if annealed_guidance:
                        g = (i + 1) / (T - 1) * guidance
                    else:
                        g = guidance
                    if 'a' in guide_what:
                        xa = xa + g * (xa - xa_u)
                    if 'c' in guide_what:
                        xc = xc + g * (xc - xc_u)
                alpha = xa[:, 0].float().exp().type(xa.dtype)  # get rid of the sequence dimension
                x_next[:, i + 1] = z[:, i + 1] * alpha + xc[:, 0]
                if show_trace:
                    iter_trace = None

        elif mode == 'J':
            x_curr = x_next
            n_iter = 0
            diff = 1e6
            iter_trace = []
            while (n_iter < max_jacobi) and (diff >ebound):
                x_next = self.proj_in(x_curr) + pos_embed
                xa, xc = self.reverse_substep(x_next, y, jacobi_attn_mask=self.attn_mask)
                if guidance > 0 and guide_what:
                    xa_u, xc_u = self.reverse_substep(x_next, None, jacobi_attn_mask=self.attn_mask, attn_temp=attn_temp)
                    if annealed_guidance:
                        g = torch.arange(1,T+1,device=z.device) / (T - 1) * guidance
                        g = g.view(1,len(g),1)
                    else:
                        g = guidance
                    if 'a' in guide_what:
                        xa = xa + g * (xa - xa_u)   
                    if 'c' in guide_what:
                        xc = xc + g * (xc - xc_u)
                xa = torch.cat([torch.zeros_like(xa[:, :1]), xa[:, :-1]], dim=1)
                xc = torch.cat([torch.zeros_like(xc[:, :1]), xc[:, :-1]], dim=1)
                alpha = xa.float().exp().type(xa.dtype)
                gamma = xc
                x_next = alpha * z + gamma
                x_next = torch.clamp(x_next, min=-3, max=3)

                diff = torch.norm(x_next - x_curr) / (B * C)
                n_iter = n_iter + 1
                x_curr = x_next
                if show_trace:
                    iter_trace.append((torch.norm(X_target - x_next).item())/(B * T * C))
                    # print(n_iter)

        elif mode == 'GSJ':
            base_attn_mask = torch.tril(torch.ones(jacobi_size, jacobi_size,device=z.device))
            iter_trace = []

            for i in range(num_GS):
                jacobi_attn_mask = torch.cat([torch.ones(jacobi_size, jacobi_size,device=z.device)]*i + [base_attn_mask], dim=1)
                last = (i == num_GS - 1)
                if last:
                    z_sub = z[:,-(jacobi_size-1):]
                    pos_embed_sub = pos_embed[-jacobi_size:-1]
                    x_curr_sub = x_next[:,-(jacobi_size-1):].clone()
                    jacobi_attn_mask = jacobi_attn_mask[:-1,:-1]
                else:
                    z_sub = z[:,(i*jacobi_size+1):((i+1)*jacobi_size+1)]
                    pos_embed_sub = pos_embed[(i*jacobi_size):((i+1)*jacobi_size)]
                    x_curr_sub = x_next[:,(i*jacobi_size+1):((i+1)*jacobi_size+1)].clone()
                
                if incre1 and i == 0:
                    for j in range(jacobi_size):
                        x_in = x_next[:, i : i + 1]
                        x = self.proj_in(x_in) + pos_embed[i : i + 1]
                        xa, xc = self.reverse_substep(x, y, jacobi_attn_mask=None,which_cache='cond')
                        if guidance > 0 and guide_what:
                            xa_u, xc_u = self.reverse_substep(x, None, jacobi_attn_mask=None, attn_temp=attn_temp, which_cache='uncond')
                            if annealed_guidance:
                                g = (i + 1) / (T - 1) * guidance
                            else:
                                g = guidance
                            if 'a' in guide_what:
                                xa = xa + g * (xa - xa_u)
                            if 'c' in guide_what:
                                xc = xc + g * (xc - xc_u)
                        alpha = xa[:, 0].float().exp().type(xa.dtype)  # get rid of the sequence dimension
                        x_next[:, i + 1] = z[:, i + 1] * alpha + xc[:, 0]
                        self.cat_kv_temp('cond')
                        if guidance > 0 and guide_what:
                            self.cat_kv_temp('uncond')
                    continue

                n_iter = 0
                diff = 1e6
                iter_trace_module = []

                while (n_iter < max_jacobi) and (diff > ebound):
                    if last:
                        x_next_sub = x_next[:,-jacobi_size:-1].clone()
                    else:
                        x_next_sub = x_next[:,(i*jacobi_size):((i+1)*jacobi_size)].clone()                
                    x_next_sub = self.proj_in(x_next_sub) + pos_embed_sub
                    
                    xa, xc = self.reverse_substep(x_next_sub,y,jacobi_attn_mask,which_cache='cond')
                    if guidance > 0 and guide_what:
                        xa_u, xc_u = self.reverse_substep(x_next_sub,None,jacobi_attn_mask,attn_temp,which_cache='uncond')
                        if annealed_guidance:
                            if last:
                                g = torch.arange(T - jacobi_size + 1 ,T,device=z.device) / (T - 1) * guidance
                            else:
                                g = torch.arange(i*jacobi_size+1, (i+1)*jacobi_size+1,device=z.device) / (T - 1) * guidance
                            g = g.view(1,len(g),1)
                        else:
                            g = guidance
                        if 'a' in guide_what:
                            xa = xa + g * (xa - xa_u)   
                        if 'c' in guide_what:
                            xc = xc + g * (xc - xc_u)
                    alpha = xa.float().exp().type(xa.dtype)
                    gamma = xc
                    x_next_sub = alpha * z_sub + gamma
                    x_next_sub = torch.clamp(x_next_sub, min=-3, max=3)

                    if last:
                        x_next[:,-(jacobi_size-1):] = x_next_sub
                        if show_trace:
                            iter_trace_module.append((torch.norm(X_target[:,-(jacobi_size-1):] - x_next_sub).item())/(B * x_next_sub.size(1) * C))
                            print(n_iter)
                    else:
                        x_next[:,(i*jacobi_size+1):((i+1)*jacobi_size+1)] = x_next_sub
                        if show_trace:
                            iter_trace_module.append((torch.norm(X_target[:,(i*jacobi_size+1):((i+1)*jacobi_size+1)] - x_next_sub).item())/(B * x_next_sub.size(1) * C))
                            print(n_iter)

                    diff = torch.norm(x_next_sub - x_curr_sub) / (B * C)
                    n_iter = n_iter + 1
                    x_curr_sub = x_next_sub
                if show_trace:
                    iter_trace.append(iter_trace_module)
                self.cat_kv_temp('cond')
                if guidance > 0 and guide_what:
                    self.cat_kv_temp('uncond')

        self.set_GSJmode('J')
        if show_trace:
            return self.permutation(x_next, inverse=True), iter_trace
        else:
            return self.permutation(x_next, inverse=True)


class Model(torch.nn.Module):
    VAR_LR: float = 0.1
    var: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        img_size: int,
        patch_size: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        num_classes: int = 0,
        norm: int = 2,
        detect_mode: bool = False,
    ):
        super().__init__()
        self.num_blocks = num_blocks
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.detect_mode = detect_mode
        self.norm = norm
        permutations = [PermutationIdentity(self.num_patches), PermutationFlip(self.num_patches)]

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    in_channels * patch_size**2,
                    channels,
                    self.num_patches,
                    permutations[i % 2],
                    layers_per_block,
                    num_classes=num_classes,
                    detect_mode = self.detect_mode,
                    norm = self.norm,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

        self.register_buffer('var', torch.ones(self.num_patches, in_channels * patch_size**2))

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert an image (N,C',H,W) to a sequence of patches (N,T,C')"""
        u = torch.nn.functional.unfold(x, self.patch_size, stride=self.patch_size)
        return u.transpose(1, 2)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a sequence of patches (N,T,C) to an image (N,C',H,W)"""
        u = x.transpose(1, 2)
        return torch.nn.functional.fold(u, (self.img_size, self.img_size), self.patch_size, stride=self.patch_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        x = self.patchify(x)
        outputs = []
        logdets = torch.zeros((), device=x.device)
        if self.detect_mode:
            IGN_blocks = []
            CRN_blocks = []
            for block in self.blocks:
                x, logdet, IGN, CRN = block(x, y)
                logdets = logdets + logdet
                outputs.append(x)
                IGN_blocks.append(IGN)
                CRN_blocks.append(CRN)
            return x, outputs, logdets, IGN_blocks, CRN_blocks
        else:
            for block in self.blocks:
                x, logdet = block(x, y)
                logdets = logdets + logdet
                outputs.append(x)
            return x, outputs, logdets

    def update_prior(self, z: torch.Tensor):
        z2 = (z**2).mean(dim=0)
        self.var.lerp_(z2.detach(), weight=self.VAR_LR)

    def get_loss(self, z: torch.Tensor, logdets: torch.Tensor):
        return 0.5 * z.pow(2).mean() - logdets.mean()

    def reverse(
        self,
        x: torch.Tensor,
        y: torch.Tensor | None = None,
        guidance: float = 0,
        guide_what: str = 'ac',
        attn_temp: float = 1.0,
        annealed_guidance: bool = False,
        return_sequence: bool = False,
        num_GS_list: list = None,
        max_jacobi_list: list = None,
        guess_list: list = None,
        ebound: float = 1e-6,
        show_trace: bool = False,
        X_target_list: list = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        
        if num_GS_list is None:
            num_GS_list = [0] * self.num_blocks
        if max_jacobi_list is None:
            max_jacobi_list = [1] * self.num_blocks
        if guess_list is None:
            guess_list = [0] * self.num_blocks

        seq = [self.unpatchify(x)]
        x = x * self.var.sqrt()

        if show_trace:
            iter_trace_blocks = []
            for i,block in enumerate(reversed(self.blocks)):
                x, iter_trace = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance,show_trace=show_trace,
                                num_GS=num_GS_list[i],max_jacobi=max_jacobi_list[i],zero_guess=guess_list[i],ebound=ebound,X_target=X_target_list[i])
                seq.append(self.unpatchify(x))
                iter_trace_blocks.append(iter_trace)
            x = self.unpatchify(x)
            return x, iter_trace_blocks
        else:
            for i,block in enumerate(reversed(self.blocks)):
                if i==7:
                    incre1 = True
                else:
                    incre1 = False
                x = block.reverse(x, y, guidance, guide_what, attn_temp, annealed_guidance,
                                num_GS=num_GS_list[i],max_jacobi=max_jacobi_list[i],zero_guess=guess_list[i],ebound=ebound,
                                incre1=incre1)
                seq.append(self.unpatchify(x))
            x = self.unpatchify(x)
            if not return_sequence:
                return x
            else:
                return seq
