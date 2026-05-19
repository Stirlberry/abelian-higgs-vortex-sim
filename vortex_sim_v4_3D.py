#!/usr/bin/env python3
"""
Abelian-Higgs Vortex Simulation  —  3D Sandbox  (v4 layout)
============================================================
Extension of vortex_sim_v4.py to three spatial dimensions.

Layout mirrors v4 exactly:
  Left  60 %  — main display (XY slice with circular mask, like v4).
                Two small insets (XZ, YZ) occupy the black corner.
                "3D GRAPH: ON" overlays a vortex-core scatter in the
                same area; the slices are hidden while it is active.
  Right 40 %  — identical control panel to v4.
                REFLECT button is replaced by "3D GRAPH: OFF / ON".

Physics:
  Same damped leapfrog, same PDG 2024 constants, same A₀ gauge coupling.
  Laplacian: 7-point numpy-roll stencil (no scipy needed).
  PML: spherical absorbing layer.
  Vortices become strings running along z.
  Spin sources rotate the phase in the XY plane, uniform along z.

Grid: N=64 cube (262 144 cells), speed slider 1–6 steps/frame.

PDG 2024: λ=0.13, v=1, sin²θW=0.231
Requirements: pip install numpy matplotlib --break-system-packages
Run:         python vortex_sim_v4_3D.py
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D   # registers 3-D projection

# ─── PDG 2024 constants ───────────────────────────────────────────────────────
V          = 1.0
LAMBDA_PDG = 0.13
SIN2_TW    = 0.23129
SIN_TW     = np.sqrt(SIN2_TW)
COS_TW     = np.sqrt(1.0 - SIN2_TW)
G_W        = 0.30
G_Z        = G_W / COS_TW          # 0.342
E_W        = G_W * SIN_TW          # 0.144
SPIN_SCALE = SIN_TW * 0.00125
OMEGA_BASE = 0.18

# ─── Grid ─────────────────────────────────────────────────────────────────────
N  = 64
CX = CY = CZ = N // 2
C2   = 0.65
DT   = 0.08
DAMP = 0.9999

CORE_THRESHOLD = 0.30

# ─── Precomputed arrays ───────────────────────────────────────────────────────
# 3-D broadcast arrays  (axis 0 = y, axis 1 = x, axis 2 = z)
_ii  = np.arange(N, dtype=np.float32)[:, None, None]   # y
_jj  = np.arange(N, dtype=np.float32)[None, :, None]   # x
_kk  = np.arange(N, dtype=np.float32)[None, None, :]   # z

# 2-D arrays for A0 and display masks
_ii2 = np.arange(N, dtype=np.float32)[:, None]         # y
_jj2 = np.arange(N, dtype=np.float32)[None, :]         # x

R_GRID_3D = np.sqrt((_ii - CY)**2 + (_jj - CX)**2 + (_kk - CZ)**2)
R_GRID_2D = np.sqrt((_ii2 - CY)**2 + (_jj2 - CX)**2)

R_ACTIVE  = int(N * 0.36)   # ≈ 23  (same fraction as v4)
R_PML_END = int(N * 0.48)   # ≈ 31
PML_MAX   = 0.55

pml = np.zeros((N, N, N), dtype=np.float32)
_s  = np.clip((R_GRID_3D - R_ACTIVE) / max(R_PML_END - R_ACTIVE, 1), 0.0, 1.0)
pml[R_GRID_3D >= R_ACTIVE] = (PML_MAX * _s**3)[R_GRID_3D >= R_ACTIVE]

# Circular display mask for the XY slice view (matches v4 style)
in_circle = R_GRID_2D <= N // 2 - 1

# ─── Fields ───────────────────────────────────────────────────────────────────
p1  = np.ones( (N, N, N), dtype=np.float32)
p2  = np.zeros((N, N, N), dtype=np.float32)
p1P = np.ones( (N, N, N), dtype=np.float32)
p2P = np.zeros((N, N, N), dtype=np.float32)
# A0 defined in XY plane (shape N×N×1) — broadcasts over z automatically
A0  = np.zeros((N, N, 1), dtype=np.float32)

# ─── UI state ─────────────────────────────────────────────────────────────────
S = dict(
    paused    = False,
    block     = 'plus',
    spin_dir  = 0,
    spin_rate = 3,
    count     = 6,
    lam       = LAMBDA_PDG,
    spf       = 1,
    show_3d   = False,
    nv        = 0,
)

spin_sources  = []
background_A0 = 0.0

def recompute_A0():
    global A0
    sig2 = 18.0 ** 2
    A0[:, :, 0] = background_A0
    for src in spin_sources:
        dr2 = (_ii2 - src['ci'])**2 + (_jj2 - src['cj'])**2
        A0[:, :, 0] += src['m'] / (1.0 + dr2 / sig2)

# ─── Laplacian (7-point np.roll stencil — no scipy needed) ───────────────────
def lap(f):
    return (np.roll(f,  1, 0) + np.roll(f, -1, 0)
          + np.roll(f,  1, 1) + np.roll(f, -1, 1)
          + np.roll(f,  1, 2) + np.roll(f, -1, 2)
          - 6.0 * f)

# ─── Physics step (damped leapfrog — same form as v4) ────────────────────────
def step():
    global p1, p2, p1P, p2P
    l1  = lap(p1);  l2 = lap(p2)
    pot = S['lam'] * (p1*p1 + p2*p2 - 1.0)
    v1  = p1 - p1P;  v2 = p2 - p2P
    dt2 = DT * DT
    # A0 shape (N,N,1) broadcasts over z
    n1  = (1+DAMP)*p1 - DAMP*p1P + dt2*(C2*l1 - pot*p1) - pml*v1 - 2.0*A0*v2
    n2  = (1+DAMP)*p2 - DAMP*p2P + dt2*(C2*l2 - pot*p2) - pml*v2 + 2.0*A0*v1
    # Dirichlet vacuum on all six faces
    for arr, val in ((n1, V), (n2, 0.0)):
        arr[ 0,:,:] = val;  arr[-1,:,:] = val
        arr[ :,0,:] = val;  arr[ :,-1,:] = val
        arr[ :,:,0] = val;  arr[ :,:,-1] = val
    p1P[:] = p1;  p2P[:] = p2
    p1[:] = n1;   p2[:] = n2

    # Continuous phase rotation in XY plane, uniform along z (same as v4)
    if spin_sources:
        sig2_ = float((R_ACTIVE * 0.60) ** 2)
        for src in spin_sources:
            r2    = ((_ii2 - src['ci'])**2 + (_jj2 - src['cj'])**2)[:, :, np.newaxis]
            angle = (src['omega'] * DT * np.exp(-r2 / sig2_)).astype(np.float32)
            cos_a = np.cos(angle);  sin_a = np.sin(angle)
            p1_r  = p1  * cos_a - p2  * sin_a
            p2[:] = p1  * sin_a + p2  * cos_a;  p1[:] = p1_r
            p1P_r  = p1P * cos_a - p2P * sin_a
            p2P[:] = p1P * sin_a + p2P * cos_a;  p1P[:] = p1P_r

# ─── Vortex string tools ──────────────────────────────────────────────────────
def place_vortex(ci, cj, charge):
    """Place a vortex string along z centred at row=ci (y), col=cj (x)."""
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
    """Tangential velocity kick in XY plane applied to all z-slices."""
    global p1P, p2P
    if S['spin_dir'] == 0:
        return
    infl  = 18;  infl2 = float(infl * infl)
    speed = S['spin_rate'] * 0.04 * S['spin_dir'] * charge
    i0 = max(2, ci-infl);  i1 = min(N-2, ci+infl)
    j0 = max(2, cj-infl);  j1 = min(N-2, cj+infl)
    if i0 >= i1 or j0 >= j1:
        return
    li = np.arange(i0, i1, dtype=np.float32)[:, None, None]
    lj = np.arange(j0, j1, dtype=np.float32)[None, :, None]
    dr = li - ci;  dc = lj - cj
    r2 = dr*dr + dc*dc;  r = np.sqrt(r2) + 0.01
    w  = np.where((r2 >= 1.0) & (r2 <= infl2),
                  np.exp(-r2 / (infl2 * 0.30)), 0.0).astype(np.float32)
    vi = -dc / r * speed * w
    vj =  dr / r * speed * w
    s1 = p1[i0-1:i1+1, j0-1:j1+1, :]
    s2 = p2[i0-1:i1+1, j0-1:j1+1, :]
    g1i = (s1[2:, 1:-1, :] - s1[:-2, 1:-1, :]) * 0.5
    g1j = (s1[1:-1, 2:, :] - s1[1:-1, :-2, :]) * 0.5
    g2i = (s2[2:, 1:-1, :] - s2[:-2, 1:-1, :]) * 0.5
    g2j = (s2[1:-1, 2:, :] - s2[1:-1, :-2, :]) * 0.5
    p1P[i0:i1, j0:j1, :] = p1[i0:i1, j0:j1, :] - DT * (vi*g1i + vj*g1j)
    p2P[i0:i1, j0:j1, :] = p2[i0:i1, j0:j1, :] - DT * (vi*g2i + vj*g2j)

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
    p1[:]=V; p2[:]=0; p1P[:]=V; p2P[:]=0
    spin_sources.clear();  A0[:]=0;  S['nv']=0
    refresh_info()

# ─── Colour map (same formula as v4) ─────────────────────────────────────────
_T3 = 2*np.pi/3

def to_rgb(s1, s2, mask=None):
    amp = np.clip(np.sqrt(s1*s1 + s2*s2), 0, 1)
    ph  = np.arctan2(s2, s1)
    r = np.clip((0.5 + 0.5*np.cos(ph))       * amp, 0, 1)
    g = np.clip((0.5 + 0.5*np.cos(ph - _T3)) * amp, 0, 1)
    b = np.clip((0.5 + 0.5*np.cos(ph + _T3)) * amp, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    if mask is not None:
        rgb[~mask] = 0
    return (rgb * 255).astype(np.uint8)

def to_rgb_main():
    return to_rgb(p1[:, :, CZ], p2[:, :, CZ], in_circle)

# ─── Figure (same figsize and proportions as v4) ──────────────────────────────
fig = plt.figure(figsize=(13, 8), facecolor='#080808')
try:
    fig.canvas.manager.set_window_title('Abelian-Higgs Vortex Simulation  v4 3D')
except Exception:
    pass

# ── Main display — XY slice with circular mask (identical rect to v4) ─────────
ax = fig.add_axes([0.01, 0.03, 0.60, 0.95])
ax.set_facecolor('#000');  ax.set_aspect('equal');  ax.axis('off')
img_h = ax.imshow(to_rgb_main(), origin='upper', extent=[0,N,N,0],
                  animated=True, interpolation='nearest')
ax.set_xlim(0, N);  ax.set_ylim(N, 0)
ax.add_patch(Circle((CX, CY), N//2-2, fill=False, edgecolor='#1a1a1a', lw=1.5))

info_txt = ax.text(CX, N-3, '', ha='center', va='bottom',
                   color='#444', fontsize=7.5, fontfamily='monospace')

def refresh_info():
    pause_str = '  [ PAUSED ]' if S['paused'] else ''
    info_txt.set_text(
        f"strings: {S['nv']}  λ={S['lam']:.3f}  sin²θW=0.231  eW=0.144{pause_str}"
    )
refresh_info()

# ── XZ and YZ insets — in the black corner below-left of the circle ───────────
# The circle's lower-left quadrant is dark; these 2 insets sit there safely.
ax_xz = fig.add_axes([0.018, 0.044, 0.148, 0.198], facecolor='#050505')
ax_yz = fig.add_axes([0.172, 0.044, 0.148, 0.198], facecolor='#050505')
for ax_s, lbl in ((ax_xz, f'XZ  y={CY}'), (ax_yz, f'YZ  x={CX}')):
    ax_s.axis('off')
    ax_s.set_title(lbl, color='#333', fontsize=6, fontfamily='monospace', pad=1)
img_xz = ax_xz.imshow(to_rgb(p1[:, CY, :], p2[:, CY, :]),
                       origin='upper', extent=[0,N,N,0],
                       animated=True, interpolation='nearest')
img_yz = ax_yz.imshow(to_rgb(p1[CX, :, :], p2[CX, :, :]),
                       origin='upper', extent=[0,N,N,0],
                       animated=True, interpolation='nearest')

# ── 3D scatter axes — same rect as main display, hidden by default ─────────────
ax3d = fig.add_axes([0.01, 0.03, 0.60, 0.95], projection='3d')
ax3d.set_xlim(0, N);  ax3d.set_ylim(0, N);  ax3d.set_zlim(0, N)
ax3d.set_xlabel('x', color='#555', fontsize=7, labelpad=1)
ax3d.set_ylabel('y', color='#555', fontsize=7, labelpad=1)
ax3d.set_zlabel('z', color='#555', fontsize=7, labelpad=1)
ax3d.tick_params(colors='#333', labelsize=5)
for pane in (ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane):
    pane.fill = False;  pane.set_edgecolor('#111')
ax3d.grid(True, color='#1a1a1a', linewidth=0.5)
ax3d.set_title(f'vortex cores  |φ| < {CORE_THRESHOLD:.2f}',
               color='#444', fontsize=8, fontfamily='monospace', pad=4)
sc_cores = ax3d.scatter([], [], [], c=[], cmap='plasma',
                        s=10, alpha=0.6, vmin=0, vmax=N, depthshade=True)
ax3d.set_visible(False)

# ─── Control panel — pixel-identical to v4 ────────────────────────────────────
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
ax_hdr.text(0.5, 0.5, 'VORTEX 3D  ·  PDG 2024', ha='center', va='center',
            color='#444', fontsize=8, fontfamily='monospace')
ax_hdr.axis('off')

# Block buttons — 2 rows × 3 (same positions as v4)
_bw = 0.111;  _bh = 0.052;  _bx0 = 0.625
_by1 = 0.868;  _by2 = 0.807

b_plus   = mkbtn([_bx0+0*(_bw+0.005), _by1, _bw, _bh], '⊕  string')
b_minus  = mkbtn([_bx0+1*(_bw+0.005), _by1, _bw, _bh], '⊖  string')
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

# Spin label + buttons (same as v4)
ax_slbl = fig.add_axes([0.625, 0.765, 0.365, 0.030], facecolor='#080808')
ax_slbl.text(0.03, 0.5, 'SPIN', va='center', color='#444',
             fontsize=8, fontfamily='monospace')
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

# Sliders (same positions and style as v4; speed capped at 6 for 3-D cost)
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
    s.valtext.set_position((0.92, 0.5));  s.valtext.set_ha('right')
    return s

sl_rate  = mk_slider(0.665, 'spin rate', 1,  8,  3,        step=1)
sl_count = mk_slider(0.607, 'count',     2, 10,  6,        step=1)
sl_lam   = mk_slider(0.549, 'λ',         0.03, 0.25, LAMBDA_PDG, step=0.01, col='#303048')
sl_spf   = mk_slider(0.491, 'speed',     1,  6,  1,        step=1)

sl_rate.on_changed( lambda v: S.update(spin_rate=int(v)))
sl_count.on_changed(lambda v: S.update(count=int(v)))
sl_lam.on_changed(  lambda v: [S.update(lam=float(v)), refresh_info()])
sl_spf.on_changed(  lambda v: S.update(spf=int(v)))

# Action buttons — REFLECT replaced by 3D GRAPH toggle
_aw = 0.365;  _ah = 0.052
b_pause   = mkbtn([_sx, 0.420, _aw, _ah], 'PAUSE')
b_clear   = mkbtn([_sx, 0.358, _aw, _ah], 'CLEAR')
b_3d      = mkbtn([_sx, 0.296, _aw, _ah], '3D GRAPH: OFF', 8)
b_quit    = mkbtn([_sx, 0.234, _aw, _ah], 'QUIT')

def on_pause(e=None):
    S['paused'] = not S['paused']
    b_pause.label.set_text('PLAY' if S['paused'] else 'PAUSE')
    b_pause.label.set_color(CTON if S['paused'] else CT)
    refresh_info();  fig.canvas.draw_idle()

_tick = [0]   # frame counter for scatter throttle

def on_3d(e=None):
    S['show_3d'] = not S['show_3d']
    show = S['show_3d']
    b_3d.label.set_text('3D GRAPH: ON' if show else '3D GRAPH: OFF')
    b_3d.label.set_color(CTON if show else CT)
    ax.set_visible(not show)
    ax_xz.set_visible(not show)
    ax_yz.set_visible(not show)
    ax3d.set_visible(show)
    _tick[0] = 0   # force scatter refresh on next frame
    fig.canvas.draw_idle()

b_pause.on_clicked(on_pause)
b_clear.on_clicked(lambda e: do_reset())
b_3d.on_clicked(   on_3d)
b_quit.on_clicked( lambda e: (plt.close('all'), sys.exit(0)))

# Legend (same as v4)
ax_leg = fig.add_axes([_sx, 0.035, _aw, 0.175], facecolor='#080808')
for y, (txt, col) in enumerate([
    ('■ blue   = stabilises ↺ CCW',  '#4477bb'),
    ('■ orange = stabilises ↻ CW',   '#bb6622'),
    ('colour = phase angle of φ',     '#555'),
    ('dark core = vortex  |φ|→0',    '#555'),
    ('SPACE=pause  R=reset  Q=quit', '#3a3a3a'),
]):
    ax_leg.text(0.04, 0.88 - y*0.19, txt, color=col,
                fontsize=7, fontfamily='monospace', va='top')
ax_leg.axis('off')

# ─── Click / key handling ─────────────────────────────────────────────────────
def on_press(event):
    if event.inaxes is not ax or event.button not in (1, 3):
        return
    ci = max(0, min(N-1, int(round(float(event.ydata)))))
    cj = max(0, min(N-1, int(round(float(event.xdata)))))
    do_place(ci, cj)

def on_key(event):
    k = event.key
    if k == ' ':                    on_pause()
    elif k in ('r', 'R'):           do_reset()
    elif k in ('q', 'Q', 'escape'): plt.close('all'); sys.exit(0)

fig.canvas.mpl_connect('button_press_event', on_press)
fig.canvas.mpl_connect('key_press_event',    on_key)

# ─── Animation ────────────────────────────────────────────────────────────────
def update(frame):
    _tick[0] += 1
    if not S['paused']:
        for _ in range(S['spf']):
            step()

    if S['show_3d']:
        if _tick[0] % 4 == 0:
            amp3d = np.sqrt(p1*p1 + p2*p2)
            idx   = np.where(amp3d < CORE_THRESHOLD)
            n_pts = len(idx[0])
            if n_pts > 0:
                if n_pts > 3000:
                    sel = np.random.choice(n_pts, 3000, replace=False)
                    yi, xi, zi = idx[0][sel], idx[1][sel], idx[2][sel]
                else:
                    yi, xi, zi = idx[0], idx[1], idx[2]
                sc_cores._offsets3d = (xi.astype(float),
                                       yi.astype(float),
                                       zi.astype(float))
                sc_cores.set_array(zi.astype(float))
            else:
                sc_cores._offsets3d = (np.array([]), np.array([]), np.array([]))
                sc_cores.set_array(np.array([]))
    else:
        img_h.set_array(to_rgb_main())
        img_xz.set_array(to_rgb(p1[:, CY, :], p2[:, CY, :]))
        img_yz.set_array(to_rgb(p1[CX, :, :], p2[CX, :, :]))

    return []

# blit=False required while the 3-D axes exists in the figure
ani = animation.FuncAnimation(fig, update, interval=25,
                              blit=False, cache_frame_data=False)

print("Abelian-Higgs Vortex Simulation  v4 3D")
print(f"PDG 2024: λ={LAMBDA_PDG}  sin²θW={SIN2_TW}  gZ={G_Z:.3f}  eW={E_W:.3f}")
print(f"Grid {N}³  R_active={R_ACTIVE}  steps/frame 1–6")
print()
print("  Click sim           place selected block (strings along z)")
print("  3D GRAPH button     toggle vortex-core scatter view")
print("  SPACE / button      pause / resume")
print("  R                   reset field")
print("  Q / Escape          quit")
plt.show()
