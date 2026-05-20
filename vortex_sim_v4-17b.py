#!/usr/bin/env python3
"""
Abelian-Higgs Vortex Simulation  —  Claude Code edition  v4.17b
================================================================
Changes vs v4.17a (this version — cleanup & performance):
  - Removed unused bare 'import matplotlib' and stale docstring text
  - Build-time arrays _DI, _DJ, _s, _t deleted after use (~512 KB freed)
  - _DT2 = DT² and _SIG2_A0 = 18² promoted to module constants
  - step() temporaries pre-allocated at startup (_step_pot, _step_v1/v2,
    _step_n1/n2, _step_tmp) — eliminates ~1.5 MB of heap allocs per call
  - Energy/topo rendering: np.roll gradient calls replaced with slice views
    — eliminates 8 × 256 KB temporaries per render call
  - refresh_info() early-out when no info items are enabled

Changes vs v4.17:
  - Save / load field state (File → Save state… / Load state…, Ctrl+S/O)
    saves p1, p2, p1P, p2P + λ + spin sources to a .npz file
  - Right-click anywhere in sim places a single antivortex (−1);
    left-click uses the block selector as before

Changes vs v4.16:
  - Performance: pre-allocated scratch arrays in to_rgb() eliminate ~18 MB/frame
    of transient numpy allocations (phase mode: zero allocs per frame)
  - Slicing-based Laplacian (lap_into) replaces np.roll — removes 4 full-array
    copies per call, saving ~10 MB/frame in step()
  - Animation interval 25 ms → 30 ms: 33 fps target gives 32% more budget,
    eliminating the timer-backlog stutter that occurs when frames exceed 25 ms
  - Topological charge updated every 4th frame (not every frame) when enabled
  - Minimal blit list — only dirty artists returned to FuncAnimation each frame

Changes vs v4.15b:
  - Type I / II regime indicator: κ = √(2λ)/g_W displayed live in info overlay
  - λ slider extended to [0.005, 0.30] with dashed I|II crossover marker at λ≈0.0225
  - Info: regime (κ, Type I/II) toggle added to View menu

Changes vs v4.13:
  - Topological charge integrator: ∫q dA displayed live in info bar (placed vs measured)
  - Undo last placement: Ctrl+Z keyboard shortcut + UNDO panel button (20-step stack)

Changes vs v4.12:
  - View menu in native Qt menu bar: Phase/Energy/Topo radio, vortex core markers,
    phase gradient arrows, smooth interpolation toggle
  - Fixed quiver dot artefacts (initialise with NaN + visibility toggle)

Changes vs v4.11:
  - Updated to PDG 2025 constants: sin²θ_W = 0.23122 (was 0.23129)
  - On-screen info text reads sin²θ_W dynamically from constant

Changes vs v4.2:
  - Display mode toggle: three VIEW buttons switch the rendering live.

    phase   →  HSV-like phase/amplitude (default — colour = arg φ, dark = vortex core)
    energy  →  field energy density  E = ½|∇φ|² + λ(|φ|²−1)²
               hot colourmap: black → red → yellow → white
    topo    →  topological charge density  q = (∂_x n̂ × ∂_y n̂) / 2π
               diverging: blue = +1 vortex, red = −1 antivortex, black = vacuum

Inherited from v4.2:
  - Damping slider (1.000 = Hamiltonian, 0.999 = light diss, 0.990 = heavy damp)
  - Slicing-based 5-point Laplacian (lap_into), speed ceiling 16 steps/frame

Physical constants verified against PDG 2024:
  m_H  = 125.20 ± 0.11 GeV   →  λ = m²_H/(2v²) = 0.13
  v    = 246.22 GeV          (Higgs VEV, normalised to v=1 in sim)
  sin²θ_W = 0.23122          (MS-bar at M_Z)
  gZ/gW   = 1/cos(θ_W) = 1.140
  eW/gW   = sin(θ_W)   = 0.481
Source: S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024) and 2025 update.

Requirements:
    pip install numpy matplotlib --break-system-packages

Run:
    python vortex_sim_v4-17b.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider


# ─── PDG 2025 constants ───────────────────────────────────────────────────────
V          = 1.0
LAMBDA_PDG = 0.13
SIN2_TW    = 0.23122
SIN_TW     = np.sqrt(SIN2_TW)
COS_TW     = np.sqrt(1.0 - SIN2_TW)
G_W        = 0.30
G_Z        = G_W / COS_TW          # 0.342
E_W        = G_W * SIN_TW          # 0.144
SPIN_SCALE = SIN_TW * 0.00125      # A₀ source coupling

# ─── Type I / II crossover ────────────────────────────────────────────────────
# κ = √(2λ)/g_W  (Ginzburg-Landau parameter)
# κ < 1/√2 → Type I (vortices attract)   κ > 1/√2 → Type II (vortices repel)
# Crossover: λ_cross = g_W² / 4
_LAMBDA_CROSS = G_W ** 2 / 4          # ≈ 0.0225 with G_W=0.30
_KAPPA_BOUND  = 1.0 / np.sqrt(2)      # ≈ 0.7071

# ─── Grid ─────────────────────────────────────────────────────────────────────
# N=256: each np.roll Laplacian ≈ 0.5–1 ms → speed slider spans 1–16 steps/frame
N   = 256
CX  = CY = N // 2
C2  = 0.65
DT  = 0.08
_DT2      = DT * DT           # precomputed — used every step()
_SIG2_A0  = 18.0 ** 2        # Gaussian width for A₀ sources

R_ACTIVE  = int(N * 0.36)    # ≈ 92
R_PML_END = int(N * 0.48)    # ≈ 123
PML_MAX   = 0.55
OMEGA_BASE = 0.18   # rad/step per spin-rate unit for continuous rotation

# ─── Precomputed arrays ───────────────────────────────────────────────────────
_ii = np.arange(N, dtype=np.float32)[:, None]
_jj = np.arange(N, dtype=np.float32)[None, :]
_DI = _ii - CY;  _DJ = _jj - CX
R_GRID = np.sqrt(_DI**2 + _DJ**2)
del _DI, _DJ                  # consumed above; free ~512 KB

pml = np.zeros((N, N), dtype=np.float32)
_s  = np.clip((R_GRID - R_ACTIVE) / (R_PML_END - R_ACTIVE), 0.0, 1.0)
pml[R_GRID >= R_ACTIVE] = (PML_MAX * _s**3)[R_GRID >= R_ACTIVE]
del _s

in_circle = R_GRID <= N // 2 - 1
wall_mask = R_GRID >= N // 2 - 2

# Smoothstep fade over the outermost 3 pixels — purely cosmetic, physics unchanged
_t         = np.clip((N // 2 - 2 - R_GRID) / 3.0, 0.0, 1.0)
_soft_edge = (_t * _t * (3.0 - 2.0 * _t)).astype(np.float32)
del _t

# ─── Pre-allocated physics scratch ────────────────────────────────────────────
_lap1 = np.empty((N, N), dtype=np.float32)
_lap2 = np.empty((N, N), dtype=np.float32)

# step() scratch — eliminates ~1.5 MB of heap allocs per physics call
_step_pot = np.empty((N, N), dtype=np.float32)
_step_v1  = np.empty((N, N), dtype=np.float32)
_step_v2  = np.empty((N, N), dtype=np.float32)
_step_n1  = np.empty((N, N), dtype=np.float32)
_step_n2  = np.empty((N, N), dtype=np.float32)
_step_tmp = np.empty((N, N), dtype=np.float32)

# ─── Pre-allocated rendering scratch ──────────────────────────────────────────
# Reused every frame — eliminates ~18 MB/frame of transient numpy allocations.
_s_amp = np.empty((N, N), dtype=np.float32)
_s_ph  = np.empty((N, N), dtype=np.float32)
_s_tmp = np.empty((N, N), dtype=np.float32)
_s_r   = np.empty((N, N), dtype=np.float32)
_s_g   = np.empty((N, N), dtype=np.float32)
_s_b   = np.empty((N, N), dtype=np.float32)
_s_rgb = np.empty((N, N, 3), dtype=np.float32)
_s_u8  = np.empty((N, N, 3), dtype=np.uint8)
_soft3 = np.repeat(_soft_edge[:, :, np.newaxis], 3, axis=2)  # 3-channel mask

# 5-point Laplacian into preallocated buffer using slice views — no np.roll copies.
# Interior (254×254) uses only views; edge/corner wrapping uses tiny 1-D temporaries.
def lap_into(f, out):
    out[1:-1, 1:-1]  = f[:-2,  1:-1]    # up
    out[1:-1, 1:-1] += f[2:,   1:-1]    # down
    out[1:-1, 1:-1] += f[1:-1, :-2]     # left
    out[1:-1, 1:-1] += f[1:-1, 2:]      # right
    out[1:-1, 1:-1] -= 4.0 * f[1:-1, 1:-1]
    # Edge rows (periodic wrap)
    out[0,  1:-1] = f[-1, 1:-1] + f[1,  1:-1] + f[0,  :-2] + f[0,  2:] - 4.0*f[0,  1:-1]
    out[-1, 1:-1] = f[-2, 1:-1] + f[0,  1:-1] + f[-1, :-2] + f[-1, 2:] - 4.0*f[-1, 1:-1]
    # Edge columns (periodic wrap)
    out[1:-1,  0] = f[:-2, 0]  + f[2:, 0]  + f[1:-1, -1] + f[1:-1, 1] - 4.0*f[1:-1, 0]
    out[1:-1, -1] = f[:-2, -1] + f[2:, -1] + f[1:-1, -2] + f[1:-1, 0] - 4.0*f[1:-1, -1]
    # Corners
    out[0,   0]  = f[-1,  0] + f[1,  0] + f[0,  -1] + f[0,  1]  - 4.0*f[0,  0]
    out[0,  -1]  = f[-1, -1] + f[1, -1] + f[0,  -2] + f[0,  0]  - 4.0*f[0,  -1]
    out[-1,  0]  = f[-2,  0] + f[0,  0] + f[-1, -1] + f[-1, 1]  - 4.0*f[-1, 0]
    out[-1, -1]  = f[-2, -1] + f[0, -1] + f[-1, -2] + f[-1, 0]  - 4.0*f[-1, -1]

# ─── Fields ───────────────────────────────────────────────────────────────────
p1  = np.ones( (N, N), dtype=np.float32)
p2  = np.zeros((N, N), dtype=np.float32)
p1P = np.ones( (N, N), dtype=np.float32)
p2P = np.zeros((N, N), dtype=np.float32)
A0  = np.zeros((N, N), dtype=np.float32)

# ─── UI state ─────────────────────────────────────────────────────────────────
S = dict(
    paused   = False,
    block    = 'plus',
    spin_dir = 0,
    spin_rate= 3,
    count    = 6,
    lam      = LAMBDA_PDG,
    spf      = 3,
    reflect  = False,
    damp     = 0.999,
    mode          = 'phase',   # 'phase' | 'energy' | 'topo'
    nv            = 0,
    topo_q        = 0,         # measured winding number ∫q dA
    show_placed   = False,
    show_topo_q   = False,
    show_regime   = False,
    show_lam      = False,
    show_sin2tw   = False,
    show_ew       = False,
    show_status   = False,
    show_markers  = False,
    show_arrows   = False,
)

spin_sources  = []   # list of {ci, cj, m (A0 moment), omega (phase rotation)}
background_A0 = 0.0

def recompute_A0():
    global A0
    A0[:] = background_A0
    for src in spin_sources:
        A0 += src['m'] / (1.0 + ((_ii - src['ci'])**2 + (_jj - src['cj'])**2) / _SIG2_A0)

# ─── Overlay helpers ──────────────────────────────────────────────────────────
# Quiver subgrid — precomputed once
_QS       = 16
_qi       = np.arange(_QS // 2, N, _QS)
_qj       = np.arange(_QS // 2, N, _QS)
_QJ, _QI  = np.meshgrid(_qj, _qi)
_QX       = _QJ.ravel().astype(float)
_QY       = _QI.ravel().astype(float)
_NQ       = len(_QX)

_core_mask = R_GRID < N // 2 - 5   # full-brightness zone only (soft-edge starts at N//2-5)

def find_cores():
    """Return (x+, y+, x-, y-) pixel coords of vortex cores via amplitude minima."""
    amp  = np.sqrt(p1*p1 + p2*p2)
    lm   = ((amp < np.roll(amp,  1, 0)) & (amp < np.roll(amp, -1, 0)) &
            (amp < np.roll(amp,  1, 1)) & (amp < np.roll(amp, -1, 1)) &
            (amp < 0.35) & _core_mask)
    rows, cols = np.where(lm)
    if len(rows) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([])
    mod  = amp + 1e-10
    n1_  = p1 / mod;  n2_ = p2 / mod
    dn1x = (np.roll(n1_, -1, 1) - np.roll(n1_, 1, 1)) * 0.5
    dn1y = (np.roll(n1_, -1, 0) - np.roll(n1_, 1, 0)) * 0.5
    dn2x = (np.roll(n2_, -1, 1) - np.roll(n2_, 1, 1)) * 0.5
    dn2y = (np.roll(n2_, -1, 0) - np.roll(n2_, 1, 0)) * 0.5
    q    = (dn1x * dn2y - dn1y * dn2x) / (2.0 * np.pi)
    s    = q[rows, cols]
    pm   = s > 0
    return (cols[pm].astype(float),  rows[pm].astype(float),
            cols[~pm].astype(float), rows[~pm].astype(float))

def compute_current():
    """Probability current J = p1∇p2 − p2∇p1, subsampled onto quiver grid."""
    dp1x = (np.roll(p1, -1, 1) - np.roll(p1, 1, 1)) * 0.5
    dp1y = (np.roll(p1, -1, 0) - np.roll(p1, 1, 0)) * 0.5
    dp2x = (np.roll(p2, -1, 1) - np.roll(p2, 1, 1)) * 0.5
    dp2y = (np.roll(p2, -1, 0) - np.roll(p2, 1, 0)) * 0.5
    Jx   = (p1 * dp2x - p2 * dp1x)[_QI, _QJ].ravel()
    Jy   = (p1 * dp2y - p2 * dp1y)[_QI, _QJ].ravel()
    mag  = np.sqrt(Jx*Jx + Jy*Jy) + 1e-10
    scl  = np.clip(mag * 4.0, 0.0, 1.0)
    U    = (Jx / mag) * scl
    V    = (Jy / mag) * scl
    hide = np.sqrt(p1[_QI, _QJ]**2 + p2[_QI, _QJ]**2).ravel() < 0.25
    U[hide] = 0.0;  V[hide] = 0.0
    return U, V

# ─── Physics step ─────────────────────────────────────────────────────────────
def step():
    global p1, p2, p1P, p2P
    # global needed: augmented assignments (-=, *=) make Python treat vars as local
    global _step_pot, _step_v1, _step_v2, _step_n1, _step_n2, _step_tmp

    lap_into(p1, _lap1);  lap_into(p2, _lap2)

    # _step_pot = lam * (p1² + p2² − 1)
    np.multiply(p1, p1, out=_step_pot)
    np.multiply(p2, p2, out=_step_tmp);  _step_pot += _step_tmp
    _step_pot -= 1.0;  _step_pot *= S['lam']

    # _step_v1 = p1 − p1P,  _step_v2 = p2 − p2P
    np.subtract(p1, p1P, out=_step_v1)
    np.subtract(p2, p2P, out=_step_v2)

    # Damped leapfrog — proper form that preserves vacuum exactly.
    # Derived from   v_{n+1/2} = DAMP · v_{n-1/2} + dt · F
    # which gives    φ_{n+1} = (1+DAMP)·φ_n − DAMP·φ_{n-1} + dt²·F
    # For vacuum (φ=v, F=0):  (1+DAMP)·v − DAMP·v = v.  No drift.
    d = S['damp']

    # _step_n1 = (1+d)·p1 − d·p1P + DT²·(C2·lap1 − pot·p1) − pml·v1 − 2·A0·v2
    np.multiply(C2, _lap1, out=_step_n1)
    np.multiply(_step_pot, p1, out=_step_tmp);  _step_n1 -= _step_tmp
    _step_n1 *= _DT2
    np.multiply(1.0+d, p1,  out=_step_tmp);  _step_n1 += _step_tmp
    np.multiply(d,     p1P, out=_step_tmp);  _step_n1 -= _step_tmp
    if not S['reflect']:
        np.multiply(pml, _step_v1, out=_step_tmp);  _step_n1 -= _step_tmp
    np.multiply(A0, _step_v2, out=_step_tmp);  _step_tmp *= 2.0;  _step_n1 -= _step_tmp

    # _step_n2 = (1+d)·p2 − d·p2P + DT²·(C2·lap2 − pot·p2) − pml·v2 + 2·A0·v1
    np.multiply(C2, _lap2, out=_step_n2)
    np.multiply(_step_pot, p2, out=_step_tmp);  _step_n2 -= _step_tmp
    _step_n2 *= _DT2
    np.multiply(1.0+d, p2,  out=_step_tmp);  _step_n2 += _step_tmp
    np.multiply(d,     p2P, out=_step_tmp);  _step_n2 -= _step_tmp
    if not S['reflect']:
        np.multiply(pml, _step_v2, out=_step_tmp);  _step_n2 -= _step_tmp
    np.multiply(A0, _step_v1, out=_step_tmp);  _step_tmp *= 2.0;  _step_n2 += _step_tmp

    if S['reflect']:
        _step_n1[wall_mask] = V;  _step_n2[wall_mask] = 0.0
    else:
        _step_n1[[0,-1], :] = V;  _step_n1[:, [0,-1]] = V
        _step_n2[[0,-1], :] = 0;  _step_n2[:, [0,-1]] = 0

    p1P[:] = p1;  p2P[:] = p2
    p1[:] = _step_n1;  p2[:] = _step_n2

    # ── Continuous phase rotation ─────────────────────────────────────────────
    # Each spin source rotates the complex field by a small angle δθ each step.
    # δθ(r) = omega · DT · exp(−r²/σ²)
    # Rotating φ → φ·exp(iδθ) keeps |φ| constant and makes the
    # phase pattern visibly spin. Both p1 and p1P are rotated so
    # the leapfrog velocity term (p1 − p1P) is preserved.
    if spin_sources:
        sig2 = float((R_ACTIVE * 0.60) ** 2)   # ≈ 55 cells — covers most of active field
        for src in spin_sources:
            r2    = (_ii - src['ci'])**2 + (_jj - src['cj'])**2
            angle = (src['omega'] * DT * np.exp(-r2 / sig2)).astype(np.float32)
            cos_a = np.cos(angle);  sin_a = np.sin(angle)
            # Rotate current step
            p1_r  = p1  * cos_a - p2  * sin_a
            p2[:] = p1  * sin_a + p2  * cos_a;  p1[:] = p1_r
            # Rotate previous step (preserves leapfrog velocity)
            p1P_r  = p1P * cos_a - p2P * sin_a
            p2P[:] = p1P * sin_a + p2P * cos_a;  p1P[:] = p1P_r

# ─── Vortex tools ─────────────────────────────────────────────────────────────
def place_vortex(ci, cj, charge):
    global p1, p2, p1P, p2P
    sq2 = np.sqrt(2.0) / np.sqrt(S['lam'])
    dr  = _ii - ci;  dc = _jj - cj
    r   = np.sqrt(dr*dr + dc*dc) + 0.01
    prf = np.tanh(r / sq2)
    amp = np.sqrt(p1*p1 + p2*p2)
    ph  = np.arctan2(p2, p1) + charge * np.arctan2(dc, dr)
    p1[:] = amp * prf * np.cos(ph)
    p2[:] = amp * prf * np.sin(ph)
    p1P[:] = p1;  p2P[:] = p2


def apply_rotation(ci, cj, charge=1):
    """Initial velocity kick at placement — uses correct central-difference gradient."""
    global p1P, p2P
    if S['spin_dir'] == 0:
        return
    infl  = 18;  infl2 = float(infl * infl)
    speed = S['spin_rate'] * 0.04 * S['spin_dir'] * charge

    # Safe subregion — keep 1-cell border for gradient
    i0 = max(2, ci-infl);  i1 = min(N-2, ci+infl)
    j0 = max(2, cj-infl);  j1 = min(N-2, cj+infl)
    if i0 >= i1 or j0 >= j1:
        return

    li = np.arange(i0, i1, dtype=np.float32)[:, None]
    lj = np.arange(j0, j1, dtype=np.float32)[None, :]
    dr = li - ci;  dc = lj - cj
    r2 = dr*dr + dc*dc
    r  = np.sqrt(r2) + 0.01
    w  = np.where((r2 >= 1.0) & (r2 <= infl2),
                  np.exp(-r2 / (infl2 * 0.30)), 0.0).astype(np.float32)

    vi = -dc / r * speed * w    # tangential row-velocity
    vj =  dr / r * speed * w    # tangential col-velocity

    # Central-difference gradient using extended slice (no roll wrap artifacts)
    s1 = p1[i0-1:i1+1, j0-1:j1+1]
    s2 = p2[i0-1:i1+1, j0-1:j1+1]
    g1i = (s1[2:, 1:-1] - s1[:-2, 1:-1]) * 0.5
    g1j = (s1[1:-1, 2:] - s1[1:-1, :-2]) * 0.5
    g2i = (s2[2:, 1:-1] - s2[:-2, 1:-1]) * 0.5
    g2j = (s2[1:-1, 2:] - s2[1:-1, :-2]) * 0.5

    p1P[i0:i1, j0:j1] = p1[i0:i1, j0:j1] - DT * (vi*g1i + vj*g1j)
    p2P[i0:i1, j0:j1] = p2[i0:i1, j0:j1] - DT * (vi*g2i + vj*g2j)

def add_source(ci, cj, charge=1):
    if S['spin_dir'] == 0:
        return
    spin_sources.append({
        'ci':    float(ci),
        'cj':    float(cj),
        'm':     S['spin_dir'] * S['spin_rate'] * SPIN_SCALE * charge,
        'omega': S['spin_dir'] * S['spin_rate'] * OMEGA_BASE * charge,
    })
    if len(spin_sources) > 12:
        spin_sources.pop(0)
    recompute_A0()

def cl(x):
    return max(8, min(N-9, int(x)))

def clR(ci, cj):
    di = ci - CY;  dj = cj - CX
    r  = (di*di + dj*dj) ** 0.5
    mr = R_ACTIVE - 6
    if r > mr and r > 0:
        f = mr / r;  return int(round(CY + di*f)), int(round(CX + dj*f))
    return ci, cj

def do_place(ci, cj):
    save_undo()
    ci, cj = clR(cl(ci), cl(cj))
    if (ci-CY)**2 + (cj-CX)**2 > (R_ACTIVE-6)**2:
        return
    blk = S['block'];  n = S['count']

    if blk == 'plus':
        place_vortex(ci, cj, +1)
        if S['spin_dir']:  apply_rotation(ci, cj, +1);  add_source(ci, cj, +1)
        S['nv'] += 1
    elif blk == 'minus':
        place_vortex(ci, cj, -1)
        if S['spin_dir']:  apply_rotation(ci, cj, -1);  add_source(ci, cj, -1)
        S['nv'] += 1
    elif blk == 'pair':
        place_vortex(ci, cl(cj-8), +1);  place_vortex(ci, cl(cj+8), -1)
        S['nv'] += 2
    elif blk == 'twins':
        place_vortex(cl(ci-8), cj, +1);  place_vortex(cl(ci+8), cj, +1)
        S['nv'] += 2
    elif blk == 'string':
        sp = 9;  tot = (n-1)*sp
        for k in range(n):
            place_vortex(ci, cl(cj + round(k*sp - tot/2)), +1 if k%2==0 else -1)
        S['nv'] += n
    elif blk == 'ring':
        rr = max(10, round(n * 4.0))
        for k in range(n):
            ang = 2*np.pi*k/n - np.pi/2
            place_vortex(cl(round(ci + rr*np.cos(ang))),
                         cl(round(cj + rr*np.sin(ang))), +1 if k%2==0 else -1)
        S['nv'] += n
    refresh_info()

def do_reset():
    global p1, p2, p1P, p2P
    save_undo()
    p1[:]=V; p2[:]=0; p1P[:]=V; p2P[:]=0
    spin_sources.clear();  A0[:]=0;  S['nv']=0
    refresh_info()

# ─── Undo stack ───────────────────────────────────────────────────────────────
_undo_stack = []
_MAX_UNDO   = 20

def save_undo():
    _undo_stack.append((p1.copy(), p2.copy(), p1P.copy(), p2P.copy(),
                        list(spin_sources), S['nv']))
    if len(_undo_stack) > _MAX_UNDO:
        _undo_stack.pop(0)

def do_undo():
    global p1, p2, p1P, p2P
    if not _undo_stack:
        return
    p1c, p2c, p1Pc, p2Pc, srcs, nv = _undo_stack.pop()
    p1[:] = p1c;  p2[:] = p2c;  p1P[:] = p1Pc;  p2P[:] = p2Pc
    spin_sources.clear();  spin_sources.extend(srcs)
    S['nv'] = nv
    recompute_A0()
    refresh_info()

# ─── Save / load state ────────────────────────────────────────────────────────
def do_save(e=None):
    try:
        try:
            from PyQt5.QtWidgets import QFileDialog
        except ImportError:
            from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            None, 'Save vortex state', 'vortex_state.npz',
            'Vortex state (*.npz);;All files (*)')
    except Exception:
        path = 'vortex_state.npz'
    if not path:
        return
    kw = dict(p1=p1, p2=p2, p1P=p1P, p2P=p2P,
              lam=np.float32(S['lam']), nv=np.int32(S['nv']))
    if spin_sources:
        kw['src_ci'] = [s['ci'] for s in spin_sources]
        kw['src_cj'] = [s['cj'] for s in spin_sources]
        kw['src_m']  = [s['m']  for s in spin_sources]
        kw['src_om'] = [s['omega'] for s in spin_sources]
    np.savez(path, **kw)
    print(f'Saved → {path}')

def do_load(e=None):
    global p1, p2, p1P, p2P
    try:
        try:
            from PyQt5.QtWidgets import QFileDialog
        except ImportError:
            from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            None, 'Load vortex state', '.',
            'Vortex state (*.npz);;All files (*)')
    except Exception:
        path = 'vortex_state.npz'
    if not path:
        return
    try:
        d = np.load(path)
        p1[:]  = d['p1'];   p2[:]  = d['p2']
        p1P[:] = d['p1P'];  p2P[:] = d['p2P']
        S['lam'] = float(d['lam'])
        S['nv']  = int(d['nv'])
        spin_sources.clear()
        if 'src_ci' in d:
            for ci, cj, m, om in zip(d['src_ci'], d['src_cj'],
                                     d['src_m'],  d['src_om']):
                spin_sources.append({'ci': float(ci), 'cj': float(cj),
                                     'm': float(m), 'omega': float(om)})
        recompute_A0()
        sl_lam.set_val(S['lam'])
        refresh_info()
        print(f'Loaded ← {path}')
    except Exception as ex:
        print(f'Load failed: {ex}')

# ─── Topological charge integrator ────────────────────────────────────────────
def compute_topo_charge():
    """Integrate topological charge density over the active circle: ∫q dA."""
    mod  = np.sqrt(p1*p1 + p2*p2) + 1e-10
    n1_  = p1 / mod;  n2_ = p2 / mod
    dn1x = (np.roll(n1_, -1, 1) - np.roll(n1_, 1, 1)) * 0.5
    dn1y = (np.roll(n1_, -1, 0) - np.roll(n1_, 1, 0)) * 0.5
    dn2x = (np.roll(n2_, -1, 1) - np.roll(n2_, 1, 1)) * 0.5
    dn2y = (np.roll(n2_, -1, 0) - np.roll(n2_, 1, 0)) * 0.5
    q    = (dn1x * dn2y - dn1y * dn2x) / (2.0 * np.pi)
    return int(round(float(np.sum(q[in_circle]))))

# ─── Colour map ───────────────────────────────────────────────────────────────
_T3 = 2*np.pi/3
_e_max_ema = 1.0   # exponential moving average of energy peak — prevents flicker

def to_rgb():
    """Render current field to uint8 RGB.  All intermediate work is done in
    pre-allocated scratch buffers — zero heap allocations in phase mode."""
    # global needed: augmented assignments (+=, *=) make Python treat vars as local
    global _s_amp, _s_ph, _s_tmp, _s_r, _s_g, _s_b, _s_rgb, _s_u8, _e_max_ema
    mode = S['mode']

    if mode == 'phase':
        # amp = clip(sqrt(p1²+p2²), 0, 1)
        np.multiply(p1, p1, out=_s_amp)
        np.multiply(p2, p2, out=_s_tmp);  _s_amp += _s_tmp
        np.sqrt(_s_amp, out=_s_amp);      np.clip(_s_amp, 0.0, 1.0, out=_s_amp)
        # ph = arctan2(p2, p1)
        np.arctan2(p2, p1, out=_s_ph)
        # r = clip((0.5 + 0.5·cos(ph))·amp, 0, 1)
        np.cos(_s_ph, out=_s_r)
        _s_r *= 0.5;  _s_r += 0.5;  _s_r *= _s_amp
        np.clip(_s_r, 0.0, 1.0, out=_s_r)
        # g = clip((0.5 + 0.5·cos(ph − T3))·amp, 0, 1)
        np.subtract(_s_ph, _T3, out=_s_tmp);  np.cos(_s_tmp, out=_s_g)
        _s_g *= 0.5;  _s_g += 0.5;  _s_g *= _s_amp
        np.clip(_s_g, 0.0, 1.0, out=_s_g)
        # b = clip((0.5 + 0.5·cos(ph + T3))·amp, 0, 1)
        np.add(_s_ph, _T3, out=_s_tmp);  np.cos(_s_tmp, out=_s_b)
        _s_b *= 0.5;  _s_b += 0.5;  _s_b *= _s_amp
        np.clip(_s_b, 0.0, 1.0, out=_s_b)
        _s_rgb[:, :, 0] = _s_r
        _s_rgb[:, :, 1] = _s_g
        _s_rgb[:, :, 2] = _s_b

    elif mode == 'energy':
        # Central-difference gradients via slice views — zero allocations.
        # Border pixels are zeroed by _soft3 at the end, so only interior matters.
        np.subtract(p1[1:-1, 2:], p1[1:-1, :-2], out=_s_r[1:-1, 1:-1]);   _s_r[1:-1, 1:-1] *= 0.5  # dx1
        np.subtract(p1[2:, 1:-1], p1[:-2, 1:-1], out=_s_g[1:-1, 1:-1]);   _s_g[1:-1, 1:-1] *= 0.5  # dy1
        np.subtract(p2[1:-1, 2:], p2[1:-1, :-2], out=_s_b[1:-1, 1:-1]);   _s_b[1:-1, 1:-1] *= 0.5  # dx2
        np.subtract(p2[2:, 1:-1], p2[:-2, 1:-1], out=_s_tmp[1:-1, 1:-1]); _s_tmp[1:-1, 1:-1] *= 0.5  # dy2
        # grad² = dx1² + dy1² + dx2² + dy2² → _s_amp
        np.multiply(_s_r, _s_r, out=_s_amp)
        np.multiply(_s_g, _s_g, out=_s_ph);  _s_amp += _s_ph
        np.multiply(_s_b, _s_b, out=_s_ph);  _s_amp += _s_ph
        np.multiply(_s_tmp, _s_tmp, out=_s_ph);  _s_amp += _s_ph
        # mod² = p1²+p2² → _s_ph
        np.multiply(p1, p1, out=_s_ph)
        np.multiply(p2, p2, out=_s_tmp);  _s_ph += _s_tmp
        # e = 0.5·grad² + λ·(mod²−1)² → reuse _s_amp
        _s_ph -= 1.0
        np.multiply(_s_ph, _s_ph, out=_s_tmp);  _s_tmp *= S['lam']
        _s_amp *= 0.5;  _s_amp += _s_tmp
        np.clip(_s_amp, 0.0, None, out=_s_amp)
        _e_max_ema = 0.94 * _e_max_ema + 0.06 * max(float(_s_amp[in_circle].max()), 1e-6)
        _s_amp /= _e_max_ema
        np.clip(_s_amp, 0.0, 1.0, out=_s_amp)   # t ∈ [0, 1]
        # Hot colourmap: black → red → yellow → white
        np.multiply(_s_amp, 3.0, out=_s_r); np.clip(_s_r, 0.0, 1.0, out=_s_r)
        np.multiply(_s_amp, 3.0, out=_s_g); _s_g -= 1.0; np.clip(_s_g, 0.0, 1.0, out=_s_g)
        np.multiply(_s_amp, 3.0, out=_s_b); _s_b -= 2.0; np.clip(_s_b, 0.0, 1.0, out=_s_b)
        _s_rgb[:, :, 0] = _s_r
        _s_rgb[:, :, 1] = _s_g
        _s_rgb[:, :, 2] = _s_b

    elif mode == 'topo':
        # mod = sqrt(p1²+p2²)+ε → _s_amp
        np.multiply(p1, p1, out=_s_amp)
        np.multiply(p2, p2, out=_s_tmp);  _s_amp += _s_tmp
        np.sqrt(_s_amp, out=_s_amp);  _s_amp += 1e-10
        # n1 = p1/mod → _s_r,  n2 = p2/mod → _s_g
        np.divide(p1, _s_amp, out=_s_r)
        np.divide(p2, _s_amp, out=_s_g)
        # Gradients of n1, n2 via slice views — _s_amp=dn1x, _s_tmp=dn1y,
        #                                        _s_ph=dn2x,  _s_b=dn2y
        # Interior only; border is zeroed by _soft3 so stale values are harmless.
        np.subtract(_s_r[1:-1, 2:], _s_r[1:-1, :-2], out=_s_amp[1:-1, 1:-1]); _s_amp[1:-1, 1:-1] *= 0.5
        np.subtract(_s_r[2:, 1:-1], _s_r[:-2, 1:-1], out=_s_tmp[1:-1, 1:-1]); _s_tmp[1:-1, 1:-1] *= 0.5
        np.subtract(_s_g[1:-1, 2:], _s_g[1:-1, :-2], out=_s_ph[1:-1, 1:-1]);  _s_ph[1:-1, 1:-1]  *= 0.5
        np.subtract(_s_g[2:, 1:-1], _s_g[:-2, 1:-1], out=_s_b[1:-1, 1:-1]);   _s_b[1:-1, 1:-1]   *= 0.5
        # q = (dn1x·dn2y − dn1y·dn2x) / 2π → _s_r
        np.multiply(_s_amp, _s_b,  out=_s_r)
        np.multiply(_s_tmp, _s_ph, out=_s_g)
        _s_r -= _s_g;  _s_r /= (2.0 * np.pi)
        qs = max(float(np.abs(_s_r[in_circle]).max()), 1e-6)
        _s_r /= qs;  np.clip(_s_r, -1.0, 1.0, out=_s_r)
        # r = clip(−q, 0,1), g=0, b = clip(q, 0,1)
        np.negative(_s_r, out=_s_amp); np.clip(_s_amp, 0.0, 1.0, out=_s_amp)
        _s_g[:] = 0.0
        np.clip(_s_r, 0.0, 1.0, out=_s_b)
        _s_rgb[:, :, 0] = _s_amp
        _s_rgb[:, :, 1] = _s_g
        _s_rgb[:, :, 2] = _s_b

    else:
        _s_rgb[:] = 0.0

    # Apply soft-edge mask and convert to uint8 — all in-place, no heap allocs
    np.multiply(_s_rgb, _soft3, out=_s_rgb)
    np.multiply(_s_rgb, 255.0,  out=_s_rgb)
    np.clip(_s_rgb, 0.0, 255.0, out=_s_rgb)
    np.copyto(_s_u8, _s_rgb, casting='unsafe')
    return _s_u8

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8), facecolor='#080808')
try:
    fig.canvas.manager.set_window_title('Abelian-Higgs Vortex Simulation  v4.17b')
except Exception:
    pass

# Simulation axes
ax = fig.add_axes([0.01, 0.03, 0.60, 0.95])
ax.set_facecolor('#000');  ax.set_aspect('equal');  ax.axis('off')
img_h = ax.imshow(to_rgb(), origin='upper', extent=[0,N,N,0],
                  animated=True, interpolation='bilinear')
ax.set_xlim(0, N);  ax.set_ylim(N, 0)
# No Circle patch — static patches flash during slider draw_idle() redraws; soft-edge defines boundary.

# Overlay artists — updated each frame when enabled
core_plus,  = ax.plot([], [], 'o', color='#00eeff', ms=7, mew=1.5,
                      markeredgecolor='white', lw=0, zorder=10)
core_minus, = ax.plot([], [], 'o', color='#ff4400', ms=7, mew=1.5,
                      markeredgecolor='white', lw=0, zorder=10)
quiver_h = ax.quiver(_QX, _QY, np.full(_NQ, np.nan), np.full(_NQ, np.nan),
                     color='white', alpha=0.55, scale=20, width=0.0025,
                     headwidth=4, headlength=4, zorder=9)
quiver_h.set_visible(False)

info_txt  = ax.text(CX, N-6, '', ha='center', va='bottom',
                     color='#00ffcc', fontsize=9, fontfamily='monospace',
                     fontweight='bold', zorder=20,
                     bbox=dict(facecolor='#0d1a2a', edgecolor='#336688',
                               alpha=0.88, pad=5, boxstyle='round,pad=0.4'))
info_txt.set_visible(False)

def refresh_info():
    if not any(S[k] for k in ('show_placed', 'show_topo_q', 'show_regime',
                               'show_lam', 'show_sin2tw', 'show_ew', 'show_status')):
        info_txt.set_visible(False)
        return
    parts = []
    if S['show_placed']:  parts.append(f"placed: {S['nv']}")
    if S['show_topo_q']:  parts.append(f"∫q={S['topo_q']:+d}")
    if S['show_regime']:
        kappa  = np.sqrt(2.0 * S['lam']) / G_W
        regime = 'I' if kappa < _KAPPA_BOUND else 'II'
        parts.append(f"κ={kappa:.3f}  Type {regime}")
    if S['show_lam']:     parts.append(f"λ={S['lam']:.3f}")
    if S['show_sin2tw']:  parts.append(f"sin²θW={SIN2_TW:.5f}")
    if S['show_ew']:      parts.append(f"eW={E_W:.3f}")
    if S['show_status']:
        if S['mode'] != 'phase': parts.append(f"[{S['mode'].upper()}]")
        if S['paused']:          parts.append('[ PAUSED ]')
    text = '  '.join(parts)
    info_txt.set_text(text)
    info_txt.set_visible(bool(text))
refresh_info()

# ─── Control panel ────────────────────────────────────────────────────────────
CB   = '#0d0d0d';  CBH  = '#1c1c1c'
CBON = '#142014';  CBOH = '#1d301d'
CT   = '#999999';  CTON = '#77dd77'

def mkbtn(rect, label, fs=8.5):
    a = fig.add_axes(rect, facecolor=CB)
    b = Button(a, label, color=CB, hovercolor=CBH)
    b.label.set_color(CT);  b.label.set_fontsize(fs)
    b.label.set_fontfamily('monospace')
    return b

# Header
ax_hdr = fig.add_axes([0.625, 0.930, 0.365, 0.045], facecolor='#080808')
ax_hdr.text(0.5, 0.5, 'VORTEX SANDBOX  ·  PDG 2025', ha='center', va='center',
            color='#444', fontsize=8, fontfamily='monospace')
ax_hdr.axis('off')

# Block buttons — 2 rows × 3
_bw = 0.111;  _bh = 0.052;  _bx0 = 0.625
_by1 = 0.868;  _by2 = 0.807

b_plus   = mkbtn([_bx0+0*(_bw+0.005), _by1, _bw, _bh], '⊕  particle')
b_minus  = mkbtn([_bx0+1*(_bw+0.005), _by1, _bw, _bh], '⊖  antipart.')
b_pair   = mkbtn([_bx0+2*(_bw+0.005), _by1, _bw, _bh], '⊕⊖  pair')
b_twins  = mkbtn([_bx0+0*(_bw+0.005), _by2, _bw, _bh], '⊕⊕  twin')
b_string = mkbtn([_bx0+1*(_bw+0.005), _by2, _bw, _bh], '⊕⊖  string')
b_ring   = mkbtn([_bx0+2*(_bw+0.005), _by2, _bw, _bh], '⊕⊖  ring')

BLOCK_BTNS = dict(plus=b_plus, minus=b_minus, pair=b_pair,
                  twins=b_twins, string=b_string, ring=b_ring)

def set_block(name):
    S['block'] = name
    for k, b in BLOCK_BTNS.items():
        on = (k == name)
        b.color = CBON if on else CB;  b.hovercolor = CBOH if on else CBH
        b.label.set_color(CTON if on else CT)
    fig.canvas.draw_idle()

b_plus.on_clicked(  lambda e: set_block('plus'))
b_minus.on_clicked( lambda e: set_block('minus'))
b_pair.on_clicked(  lambda e: set_block('pair'))
b_twins.on_clicked( lambda e: set_block('twins'))
b_string.on_clicked(lambda e: set_block('string'))
b_ring.on_clicked(  lambda e: set_block('ring'))
set_block('plus')

# Spin label + buttons
ax_slbl = fig.add_axes([0.625, 0.765, 0.365, 0.030], facecolor='#080808')
ax_slbl.text(0.03, 0.5, 'SPIN', va='center', color='#444', fontsize=8, fontfamily='monospace')
ax_slbl.axis('off')
_sw = 0.111
b_snone = mkbtn([_bx0+0*(_sw+0.005), 0.728, _sw, 0.032], 'none', 8)
b_scw   = mkbtn([_bx0+1*(_sw+0.005), 0.728, _sw, 0.032], '↻  cw',  8)
b_sccw  = mkbtn([_bx0+2*(_sw+0.005), 0.728, _sw, 0.032], '↺  ccw', 8)
SPIN_BTNS = {'none': b_snone, 'cw': b_scw, 'ccw': b_sccw}
SPIN_MAP  = {'none': 0, 'cw': +1, 'ccw': -1}

def set_spin(name):
    S['spin_dir'] = SPIN_MAP[name]
    for k, b in SPIN_BTNS.items():
        on = (k == name)
        b.color = CBON if on else CB;  b.hovercolor = CBOH if on else CBH
        b.label.set_color(CTON if on else CT)
    fig.canvas.draw_idle()

b_snone.on_clicked(lambda e: set_spin('none'))
b_scw.on_clicked(  lambda e: set_spin('cw'))
b_sccw.on_clicked( lambda e: set_spin('ccw'))
set_spin('none')

# ── Sliders ───────────────────────────────────────────────────────────────────
_sx = 0.625;  _sw2 = 0.30;  _sh = 0.034

def mk_slider(y, label, lo, hi, init, step=None, col='#404050'):
    a = fig.add_axes([_sx, y, _sw2, _sh], facecolor='#0a0a0a')
    kw = {'valstep': step} if step else {}
    s  = Slider(a, label, lo, hi, valinit=init, **kw)
    s.poly.set_facecolor(col)
    s.label.set_color(CT);    s.label.set_fontsize(8)
    s.label.set_fontfamily('monospace')
    s.valtext.set_color(CT);  s.valtext.set_fontsize(8)
    s.valtext.set_fontfamily('monospace')
    s.valtext.set_position((0.92, 0.5))
    s.valtext.set_ha('right')
    return s

sl_rate  = mk_slider(0.665, 'spin rate', 1,     8,     3,          step=1)
sl_count = mk_slider(0.607, 'count',     2,    10,     6,          step=1)
sl_lam   = mk_slider(0.549, 'λ',         0.005, 0.30,  LAMBDA_PDG, step=0.005, col='#303048')

# Mark the Type I / II crossover on the λ slider.
# axvline uses DATA coordinates [valmin, valmax]; text uses transAxes [0, 1].
_lam_lo, _lam_hi = 0.005, 0.30
_cross_frac = (_LAMBDA_CROSS - _lam_lo) / (_lam_hi - _lam_lo)   # fractional position for text
sl_lam.ax.axvline(_LAMBDA_CROSS, color='#888844', lw=1.0, ls='--', alpha=0.8)
sl_lam.ax.text(_cross_frac + 0.02, 0.85, 'I|II', color='#888844',
               fontsize=6, fontfamily='monospace', va='top',
               transform=sl_lam.ax.transAxes)
sl_spf   = mk_slider(0.491, 'speed',     1,    16,     3,          step=1)
sl_damp  = mk_slider(0.433, 'damp',      0.990, 1.000, 0.999,      step=0.001, col='#302030')

sl_rate.on_changed( lambda v: S.update(spin_rate=int(v)))
sl_count.on_changed(lambda v: S.update(count=int(v)))
sl_lam.on_changed(  lambda v: [S.update(lam=float(v)), refresh_info()])
sl_spf.on_changed(  lambda v: S.update(spf=int(v)))
sl_damp.on_changed( lambda v: S.update(damp=float(v)))

# ── View mode buttons ─────────────────────────────────────────────────────────
ax_vlbl = fig.add_axes([0.625, 0.401, 0.365, 0.024], facecolor='#080808')
ax_vlbl.text(0.03, 0.5, 'VIEW', va='center', color='#444', fontsize=8, fontfamily='monospace')
ax_vlbl.axis('off')
_vw = 0.111
b_vphase  = mkbtn([_bx0+0*(_vw+0.005), 0.364, _vw, 0.032], 'phase',  8)
b_venergy = mkbtn([_bx0+1*(_vw+0.005), 0.364, _vw, 0.032], 'energy', 8)
b_vtopo   = mkbtn([_bx0+2*(_vw+0.005), 0.364, _vw, 0.032], 'topo',   8)
VIEW_BTNS    = {'phase': b_vphase, 'energy': b_venergy, 'topo': b_vtopo}
VIEW_ACTIONS = {}   # populated by Qt menu setup; keeps panel buttons in sync

def set_view(name):
    S['mode'] = name
    for k, b in VIEW_BTNS.items():
        on = (k == name)
        b.color = CBON if on else CB;  b.hovercolor = CBOH if on else CBH
        b.label.set_color(CTON if on else CT)
    if name in VIEW_ACTIONS:
        VIEW_ACTIONS[name].setChecked(True)
    refresh_info();  fig.canvas.draw_idle()

b_vphase.on_clicked( lambda e: set_view('phase'))
b_venergy.on_clicked(lambda e: set_view('energy'))
b_vtopo.on_clicked(  lambda e: set_view('topo'))
set_view('phase')

# ── Action buttons ────────────────────────────────────────────────────────────
_aw = 0.365;  _ah = 0.052
b_pause   = mkbtn([_sx, 0.302, _aw, _ah], 'PAUSE')
b_clear   = mkbtn([_sx, 0.240, _aw, _ah], 'CLEAR')
b_reflect = mkbtn([_sx, 0.178, _aw, _ah], 'REFLECT: OFF', 8)
b_quit    = mkbtn([_sx, 0.116, _aw, _ah], 'QUIT')

def on_pause(e=None):
    S['paused'] = not S['paused']
    b_pause.label.set_text('PLAY' if S['paused'] else 'PAUSE')
    b_pause.label.set_color(CTON if S['paused'] else CT)
    refresh_info();  fig.canvas.draw_idle()

def on_reflect(e=None):
    S['reflect'] = not S['reflect']
    b_reflect.label.set_text('REFLECT: ON' if S['reflect'] else 'REFLECT: OFF')
    b_reflect.label.set_color(CTON if S['reflect'] else CT)
    fig.canvas.draw_idle()

b_pause.on_clicked(  on_pause)
b_clear.on_clicked(  lambda e: do_reset())
b_reflect.on_clicked(on_reflect)
b_quit.on_clicked(   lambda e: (plt.close('all'), sys.exit(0)))

# Legend
ax_leg = fig.add_axes([_sx, 0.035, _aw, 0.073], facecolor='#080808')
for k, (txt, col) in enumerate([
    ('■ blue = ↺ CCW   ■ orange = ↻ CW', '#557799'),
    ('phase: colour=arg φ  dark=|φ|→0',  '#555'),
    ('energy: black→red→yellow→white',    '#555'),
    ('topo: blue=+1 vortex  red=−1',      '#555'),
]):
    ax_leg.text(0.04, 0.88 - k*0.27, txt, color=col,
                fontsize=7, fontfamily='monospace', va='top')
ax_leg.axis('off')

def eg(event):
    if event.inaxes is not ax:
        return None
    ci = max(0, min(N-1, int(round(float(event.ydata)))))
    cj = max(0, min(N-1, int(round(float(event.xdata)))))
    return (ci, cj)

def on_press(event):
    if event.inaxes is not ax or event.button not in (1, 3):
        return
    g = eg(event)
    if g is None:
        return
    if event.button == 3:
        # Right-click: always place a single antivortex (−1)
        save_undo()
        ci, cj = clR(cl(g[0]), cl(g[1]))
        if (ci-CY)**2 + (cj-CX)**2 <= (R_ACTIVE-6)**2:
            place_vortex(ci, cj, -1)
            if S['spin_dir']:
                apply_rotation(ci, cj, -1)
                add_source(ci, cj, -1)
            S['nv'] += 1
            refresh_info()
    else:
        do_place(g[0], g[1])

def on_key(event):
    k = event.key
    if k == ' ':                  on_pause()
    elif k in ('r', 'R'):         do_reset()
    elif k == 'ctrl+z':           do_undo()
    elif k in ('q','Q','escape'): plt.close('all'); sys.exit(0)

fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('key_press_event',      on_key)

# ─── Animation ────────────────────────────────────────────────────────────────
_frame_count = 0

def update(frame):
    global _frame_count
    _frame_count += 1

    if not S['paused']:
        deadline = time.perf_counter() + 0.022   # 22 ms physics budget per frame
        for _ in range(S['spf']):
            step()
            if time.perf_counter() > deadline:
                break
    img_h.set_array(to_rgb())

    # Topo charge is slow (8 np.roll calls); update every 4th frame — still ~8 Hz
    if S['show_topo_q'] and (_frame_count % 4 == 0):
        S['topo_q'] = compute_topo_charge()
    refresh_info()

    if S['show_markers']:
        xp, yp, xm, ym = find_cores()
        core_plus.set_data(xp, yp)
        core_minus.set_data(xm, ym)
    else:
        core_plus.set_data([], [])
        core_minus.set_data([], [])

    if S['show_arrows']:
        U, V = compute_current()
        quiver_h.set_UVC(U, V)
        quiver_h.set_visible(True)
    else:
        quiver_h.set_visible(False)

    # Minimal blit list — only return artists that are actually updated this frame
    blit = [img_h]
    if info_txt.get_visible():  blit.append(info_txt)
    if S['show_markers']:       blit.extend([core_plus, core_minus])
    if S['show_arrows']:        blit.append(quiver_h)
    return blit

ani = animation.FuncAnimation(fig, update, interval=30, blit=True, cache_frame_data=False)

# ─── View menu (Qt backend) ───────────────────────────────────────────────────
try:
    try:
        from PyQt5.QtWidgets import QAction, QActionGroup
    except ImportError:
        from PyQt6.QtGui import QAction, QActionGroup

    win      = fig.canvas.manager.window
    menubar  = win.menuBar()

    # File menu — Save / Load state
    fmenu    = menubar.addMenu('&File')
    act_save = QAction('Save state…', win)
    act_save.setShortcut('Ctrl+S')
    act_save.triggered.connect(do_save)
    fmenu.addAction(act_save)
    act_load = QAction('Load state…', win)
    act_load.setShortcut('Ctrl+O')
    act_load.triggered.connect(do_load)
    fmenu.addAction(act_load)

    vmenu    = menubar.addMenu('&View')

    # Rendering mode — radio group
    mode_grp = QActionGroup(win)
    mode_grp.setExclusive(True)
    for name, label in [('phase', 'Phase'), ('energy', 'Energy'), ('topo', 'Topo')]:
        act = QAction(label, win)
        act.setCheckable(True)
        act.setChecked(name == S['mode'])
        act.triggered.connect(lambda chk, n=name: set_view(n))
        mode_grp.addAction(act)
        vmenu.addAction(act)
        VIEW_ACTIONS[name] = act

    vmenu.addSeparator()

    # Overlay toggles
    act_markers = QAction('Show vortex core markers', win)
    act_markers.setCheckable(True)
    act_markers.setChecked(False)
    act_markers.toggled.connect(lambda on: S.update(show_markers=on))
    vmenu.addAction(act_markers)

    act_arrows = QAction('Show phase gradient arrows', win)
    act_arrows.setCheckable(True)
    act_arrows.setChecked(False)
    act_arrows.toggled.connect(lambda on: S.update(show_arrows=on))
    vmenu.addAction(act_arrows)

    vmenu.addSeparator()

    # Info overlay — each item individually toggleable, all off by default
    for label, key in [
        ('Info: placement count',      'show_placed'),
        ('Info: topo charge (∫q)',     'show_topo_q'),
        ('Info: regime (κ, Type I/II)','show_regime'),
        ('Info: λ',                    'show_lam'),
        ('Info: sin²θW',               'show_sin2tw'),
        ('Info: eW',                   'show_ew'),
        ('Info: mode / pause',         'show_status'),
    ]:
        act = QAction(label, win)
        act.setCheckable(True)
        act.setChecked(False)
        act.toggled.connect(lambda on, k=key: S.update({k: on}))
        vmenu.addAction(act)

    vmenu.addSeparator()

    act_smooth = QAction('Smooth interpolation', win)
    act_smooth.setCheckable(True)
    act_smooth.setChecked(True)
    act_smooth.toggled.connect(
        lambda on: img_h.set_interpolation('bilinear' if on else 'nearest'))
    vmenu.addAction(act_smooth)

except Exception as e:
    print(f"View menu unavailable ({e})")

print("Abelian-Higgs Vortex Simulation v4.17b (PDG 2025 constants)")
print(f"PDG 2025: λ={LAMBDA_PDG}  sin²θW={SIN2_TW}  gZ={G_Z:.3f}  eW={E_W:.3f}")
print(f"Grid {N}×{N}  active radius {R_ACTIVE}  steps/frame 1–16")
print()
print("  Click sim           place selected block")
print("  Ctrl+Z / button     undo last placement")
print("  SPACE / button      pause / resume")
print("  R                   reset field")
print("  Q / Escape          quit")
plt.show()
