"""
Set-encoding networks for MetaSet-DRL (cascaded channel observation; single
parameter system).

The observation object is the cascaded channel (estimate) D in C^{K x L x M},
D_k = diag(h_k) H_1 -- a passive RIS can only estimate the cascaded channel
(paper 2, Saglam et al., ICCW 2023, Eq. (5)); when sigma_e^2 = 0, Dhat = D
exact, and since D is a sufficient statistic for the sum rate, a clean cascaded
observation is equivalent in information content to perfect CSI.

Token interface: D [L, K, M, 2] (element x user x antenna x real/imag).
The encoding has three levels: scalar pairs -> (element, user, antenna)
entities -> action dimensions.
The network weights are fully decoupled from (M, L, K) (at hidden=64 the
actor+critic total 84,615 parameters).

SetActor (deterministic policy):
    per RIS element -> phi_l (unit-modulus constraint, differentiable normalization)
    per (user k, antenna m) -> g_k[m] (total power constraint, differentiable normalization)

SetCritic (value decomposition):
    Q = sum_l q_phi(elem_l, phi_l) + sum_{k,m} q_g(usr_k, ant_m, g_km) + q_global
"""


import torch
import torch.nn as nn


def _mlp(in_dim, hidden, out_dim, depth=2):
    layers, d = [], in_dim
    for _ in range(depth - 1):
        layers += [nn.Linear(d, hidden), nn.ReLU()]
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class SetActor(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.enc_pair = _mlp(2, hidden, hidden)                 # (Re,Im) scalar pair
        self.enc_elem = _mlp(hidden + 2, hidden, hidden)        # element token (incl. phi_prev)
        self.enc_user = _mlp(hidden + 2, hidden, hidden)        # user token (incl. power)
        self.enc_ant = _mlp(hidden, hidden, hidden)             # antenna token
        self.dec_phi = _mlp(hidden, hidden, 2)                  # element -> phase shift (Re,Im)
        self.dec_g = _mlp(2 * hidden, hidden, 2)                # (user,antenna) -> beamforming coefficient

    def forward(self, D, phi_prev, power_t, power_r, noise_std=0.0, generator=None):
        """
        D: [B, L, K, M, 2]; phi_prev: [B, L, 2]; power_t / power_r: [B, K].
        Returns phi [B, L, 2] (unit modulus) and G [B, K, M, 2] (power
        normalization is applied by the caller).
        """
        B, L, K, M, _ = D.shape

        pair = self.enc_pair(D)                          # [B,L,K,M,h]
        ctx_elem = pair.mean(dim=(2, 3))                 # [B,L,h] pool over (user,antenna)
        ctx_user = pair.mean(dim=(1, 3))                 # [B,K,h] pool over (element,antenna)
        ctx_ant = pair.mean(dim=(1, 2))                  # [B,M,h] pool over (element,user)

        elem = self.enc_elem(torch.cat([ctx_elem, phi_prev], dim=-1))    # [B,L,h]
        usr = self.enc_user(torch.cat([ctx_user,
                                       power_t.unsqueeze(-1),
                                       power_r.unsqueeze(-1)], dim=-1))  # [B,K,h]
        ant = self.enc_ant(ctx_ant)                                      # [B,M,h]

        phi_raw = self.dec_phi(elem)                                     # [B,L,2]
        if noise_std > 0:
            phi_raw = phi_raw + noise_std * torch.randn(
                phi_raw.shape, device=phi_raw.device, generator=generator)
        phi = phi_raw / phi_raw.norm(dim=-1, keepdim=True).clamp(min=1e-6)

        usr_exp = usr.unsqueeze(2).expand(B, K, M, self.hidden)
        ant_exp = ant.unsqueeze(1).expand(B, K, M, self.hidden)
        g = self.dec_g(torch.cat([usr_exp, ant_exp], dim=-1))            # [B,K,M,2]
        if noise_std > 0:
            g = g + noise_std * torch.randn(
                g.shape, device=g.device, generator=generator)
        return phi, g

    @staticmethod
    def normalize_power(g, power_t_W):
        """Total power constraint Tr(GG^H) = power_t_W, differentiable. g: [B,K,M,2] -> same shape."""
        cur = (g ** 2).sum(dim=(1, 2, 3), keepdim=True).clamp(min=1e-9)
        return g * torch.sqrt(torch.tensor(power_t_W, device=g.device) / cur)


class SetCritic(nn.Module):
    """Critic decomposed over set elements (input is the cascaded channel observation D)."""

    def __init__(self, hidden=64):
        super().__init__()
        self.hidden = hidden
        self.enc_pair = _mlp(2, hidden, hidden)
        self.enc_elem = _mlp(hidden + 2, hidden, hidden)
        self.enc_user = _mlp(hidden + 2, hidden, hidden)
        self.q_phi = _mlp(hidden + 2, hidden, 1)                # (element, phi_l) -> scalar
        self.q_g = _mlp(2 * hidden + 2, hidden, 1)              # (usr, ant, g_km) -> scalar
        self.q_glob = _mlp(2 * hidden, hidden, 1)               # pooled state -> scalar

    def forward(self, D, power_t, power_r, phi, G):
        B, L, K, M, _ = D.shape

        pair = self.enc_pair(D)                          # [B,L,K,M,h]
        ctx_elem = pair.mean(dim=(2, 3))                 # [B,L,h]
        ctx_user = pair.mean(dim=(1, 3))                 # [B,K,h]
        ctx_ant = pair.mean(dim=(1, 2))                  # [B,M,h]

        # ---- per-element term q_phi ----
        elem = self.enc_elem(torch.cat([ctx_elem, phi], dim=-1))         # [B,L,h]
        q_phi = self.q_phi(torch.cat([elem, phi], dim=-1))               # [B,L,1]
        q_phi = q_phi.sum(dim=1).squeeze(-1)                             # [B]

        # ---- per-(user, antenna) term q_g ----
        usr_emb = self.enc_user(torch.cat([ctx_user,
                                           power_t.unsqueeze(-1),
                                           power_r.unsqueeze(-1)], dim=-1))  # [B,K,h]
        ant_emb = ctx_ant                                                # [B,M,h]
        usr_exp = usr_emb.unsqueeze(2).expand(B, K, M, self.hidden)
        ant_exp = ant_emb.unsqueeze(1).expand(B, K, M, self.hidden)
        q_g = self.q_g(torch.cat([usr_exp, ant_exp, G], dim=-1))         # [B,K,M,1]
        q_g = q_g.sum(dim=(1, 2, 3))                                     # [B]

        # ---- global coupling term ----
        z_elem = elem.mean(dim=1)                                        # [B,h]
        z_user = usr_emb.mean(dim=1)                                     # [B,h]
        q_glob = self.q_glob(torch.cat([z_elem, z_user], dim=-1)).squeeze(-1)

        return (q_phi + q_g + q_glob).unsqueeze(-1)                      # [B,1]
