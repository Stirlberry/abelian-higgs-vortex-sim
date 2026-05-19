#!/usr/bin/env python3
"""
Abelian-Higgs Vortex Simulation  —  Claude Code edition  v3
=============================================================
Fixes vs v2:
  - Slider value text repositioned inside figure bounds
  - N=256 so speed slider spans a perceptible range (was compute-bound at 512)
  - Continuous phase rotation applied each step — spin is now clearly visible
  - apply_rotation uses proper central-difference gradient (not np.roll on subarray)

Physical constants: PDG 2024, S. Navas et al., Phys. Rev. D 110, 030001 (2024)

Requirements:
    pip install numpy matplotlib scipy --break-system-packages

Run:
    python vortex_sim_v3.py
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
from scipy.ndimage import convolve

# ─── PDG 2024 constants ───────────────────────────────────────────────────────
V          = 1.0
LAMBDA_PDG = 0.13
SIN2_TW    = 0.23129
SIN_TW     = np.sqrt(SIN2_TW)
COS_TW     = np.sqrt(1.0 - SIN2_TW)
G_W        = 0.30
G_Z        = G_W / COS_TW          # 0.342
E_W        = G_W * SIN_TW          # 0.144
SPIN_SCALE = SIN_TW * 0.00125      # A₀ source coupling

# ─── Grid ─────────────────────────────────────────────────────────────────────
# N=256: each scipy convolution ≈ 1–3 ms → speed slider spans 1–12 steps/frame
N   = 256
CX  = CY = N // 2
C2  = 0.65
DT  = 0.08
DAMP = 0.9999

R_ACTIVE  = int(N * 0.36)    # ≈ 92
R_PML_END = int(N * 0.48)    # ≈ 123
PML_MAX   = 0.55
OMEGA_BASE = 0.18   # rad/step per spin-rate unit for continuous rotation

# ─── Precomputed arrays ───────────────────────────────────────────────────────
_ii = np.arange(N, dtype=np.float32)[:, None]
_jj = np.arange(N, dtype=np.float32)[None, :]
_DI = _ii - CY;  _DJ = _jj - CX
R_GRID = np.sqrt(_DI**2 + _DJ**2)

pml = np.zeros((N, N), dtype=np.float32)
_s  = np.clip((R_GRID - R_ACTIVE) / (R_PML_END - R_ACTIVE), 0.0, 1.0)
pml[R_GRID >= R_ACTIVE] = (PML_MAX * _s**3)[R_GRID >= R_ACTIVE]

in_circle = R_GRID <= N // 2 - 1
wall_mask = R_GRID >= N // 2 - 2

# 5-point Laplacian — faster than 9-point for N=256, still accurate enough
_K5 = np.array([[0, 1, 0],
                 [1,-4, 1],
                 [0, 1, 0]], dtype=np.float32)

def lap(f):
    return convolve(f, _K5, mode='wrap')

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
    nv       = 0,
)

spin_sources  = []   # list of {ci, cj, m (A0 moment), omega (phase rotation)}
background_A0 = 0.0

def recompute_A0():
    global A0
    sig2 = 18.0 ** 2
    A0[:] = background_A0
    for src in spin_sources:
        A0 += src['m'] / (1.0 + ((_ii - src['ci'])**2 + (_jj - src['cj'])**2) / sig2)

# ─── Physics step ─────────────────────────────────────────────────────────────
def step():
    global p1, p2, p1P, p2P

    l1 = lap(p1);  l2 = lap(p2)
    pot = S['lam'] * (p1*p1 + p2*p2 - 1.0)
    v1  = p1 - p1P;  v2 = p2 - p2P
    ab  = 0.0 if S['reflect'] else pml
    dt2 = DT * DT

    n1 = (2*p1 - p1P + dt2*(C2*l1 - pot*p1)) * DAMP - ab*v1 - 2.0*A0*v2
    n2 = (2*p2 - p2P + dt2*(C2*l2 - pot*p2)) * DAMP - ab*v2 + 2.0*A0*v1

    if S['reflect']:
        n1[wall_mask] = V;  n2[wall_mask] = 0.0
    else:
        n1[[0,-1], :] = V;  n1[:, [0,-1]] = V
        n2[[0,-1], :] = 0;  n2[:, [0,-1]] = 0

    p1P[:] = p1;  p2P[:] = p2
    p1[:] = n1;   p2[:] = n2

    # ── Continuous phase rotation ─────────────────────────────────────────────
    # Each spin source rotates the complex field by a small angle δθ each step.
    # δθ(r) = omega · DT · exp(−r²/σ²)
    # Rotating φ → φ·exp(iδθ) keeps |φ| constant and makes the
    # phase pattern visibly spin. Both p1 and p1P are rotated so
    # the leapfrog velocity term (p1 − p1P) is preserved.
    if spin_sources:
        sig2 = float((N * 0.07) ** 2)   # ≈ 18 cells radius
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

def remove_vortex(ci, cj, charge):
    global p1, p2, p1P, p2P
    sq2 = np.sqrt(2.0) / np.sqrt(S['lam'])
    dr  = _ii - ci;  dc = _jj - cj
    r   = np.sqrt(dr*dr + dc*dc) + 0.01
    prf = np.maximum(0.04, np.tanh(r / sq2))
    amp = np.minimum(float(V), np.sqrt(p1*p1 + p2*p2) / prf)
    ph  = np.arctan2(p2, p1) - charge * np.arctan2(dc, dr)
    p1[:] = amp * np.cos(ph);  p2[:] = amp * np.sin(ph)
    p1P[:] = p1;  p2P[:] = p2

def find_vortex(ci, cj, radius=8):
    i0 = max(0, ci-radius);  i1 = min(N, ci+radius)
    j0 = max(0, cj-radius);  j1 = min(N, cj+radius)
    if i0 >= i1 or j0 >= j1:
        return None
    reg = np.sqrt(p1[i0:i1, j0:j1]**2 + p2[i0:i1, j0:j1]**2).copy()
    li  = np.arange(i0, i1)[:, None];  lj = np.arange(j0, j1)[None, :]
    reg[(li-ci)**2 + (lj-cj)**2 > radius**2] = 1.0
    idx = np.unravel_index(np.argmin(reg), reg.shape)
    return (i0+idx[0], j0+idx[1]) if reg[idx] < 0.35 else None

def winding(ci, cj, r=4, npts=16):
    ang = np.linspace(0, 2*np.pi, npts, endpoint=False)
    si  = np.clip(np.round(ci + r*np.sin(ang)).astype(int), 0, N-1)
    sj  = np.clip(np.round(cj + r*np.cos(ang)).astype(int), 0, N-1)
    ph  = np.arctan2(p2[si, sj], p1[si, sj])
    d   = np.diff(np.append(ph, ph[0]))
    d   = (d + np.pi) % (2*np.pi) - np.pi
    return int(np.round(d.sum() / (2*np.pi)))

def apply_rotation(ci, cj):
    """Initial velocity kick at placement — uses correct central-difference gradient."""
    global p1P, p2P
    if S['spin_dir'] == 0:
        return
    infl  = 18;  infl2 = float(infl * infl)
    speed = S['spin_rate'] * 0.04 * S['spin_dir']

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

def add_source(ci, cj):
    if S['spin_dir'] == 0:
        return
    spin_sources.append({
        'ci':    float(ci),
        'cj':    float(cj),
        'm':     S['spin_dir'] * S['spin_rate'] * SPIN_SCALE,    # A₀ gauge moment
        'omega': S['spin_dir'] * S['spin_rate'] * OMEGA_BASE,    # phase rotation rate
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
        if S['spin_dir']:  apply_rotation(ci, cj);  add_source(ci, cj)
        S['nv'] += 1
    elif blk == 'minus':
        place_vortex(ci, cj, -1)
        if S['spin_dir']:  apply_rotation(ci, cj);  add_source(ci, cj)
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

# ─── Colour map ───────────────────────────────────────────────────────────────
_T3 = 2*np.pi/3

def to_rgb():
    amp = np.clip(np.sqrt(p1*p1 + p2*p2), 0, 1)
    ph  = np.arctan2(p2, p1)
    r = np.clip((0.5 + 0.5*np.cos(ph))       * amp, 0, 1)
    g = np.clip((0.5 + 0.5*np.cos(ph - _T3)) * amp, 0, 1)
    b = np.clip((0.5 + 0.5*np.cos(ph + _T3)) * amp, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~in_circle] = 0
    return (rgb * 255).astype(np.uint8)

# ─── Figure ───────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 8), facecolor='#080808')
try:
    fig.canvas.manager.set_window_title('Abelian-Higgs Vortex Simulation  v3')
except Exception:
    pass

# Simulation axes
ax = fig.add_axes([0.01, 0.03, 0.60, 0.95])
ax.set_facecolor('#000');  ax.set_aspect('equal');  ax.axis('off')
img_h = ax.imshow(to_rgb(), origin='upper', extent=[0,N,N,0],
                  animated=True, interpolation='nearest')
ax.set_xlim(0, N);  ax.set_ylim(N, 0)
ax.add_patch(Circle((CX, CY), N//2-2, fill=False, edgecolor='#1a1a1a', lw=1.5))

drag_circ = Circle((CX,CY), 6, fill=True, facecolor=(1,1,1,0.12),
                    edgecolor='white', lw=1.5, alpha=0, zorder=10)
ax.add_patch(drag_circ)
drag_lbl  = ax.text(CX, CY, '', ha='center', va='center',
                     color='white', fontsize=11, fontweight='bold', alpha=0, zorder=11)
drag_ln,  = ax.plot([], [], '--', color='white', alpha=0.3, lw=1, zorder=9)
info_txt  = ax.text(CX, N-3, '', ha='center', va='bottom',
                     color='#444', fontsize=7.5, fontfamily='monospace')

def refresh_info():
    pause_str = '  [ PAUSED ]' if S['paused'] else ''
    info_txt.set_text(
        f"vortices: {S['nv']}  λ={S['lam']:.3f}  sin²θW=0.231  eW=0.144{pause_str}"
    )
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
ax_hdr.text(0.5, 0.5, 'VORTEX SANDBOX  ·  PDG 2024', ha='center', va='center',
            color='#444', fontsize=8, fontfamily='monospace')
ax_hdr.axis('off')

# Block buttons — 2 rows × 3
_bw = 0.111;  _bh = 0.052;  _bx0 = 0.625
_by1 = 0.868;  _by2 = 0.807   # row y-bottoms

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
SPIN_MAP  = {'none': 0, 'cw': -1, 'ccw': +1}

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

# ── Sliders ──────────────────────────────────────────────────────────────────
# Width 0.30 keeps valtext within figure.  valtext repositioned to (0.92, 0.5)
# inside axes so it never clips at the right margin.
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
    # ── FIX: move valtext to left of right edge so it stays inside the figure ──
    s.valtext.set_position((0.92, 0.5))
    s.valtext.set_ha('right')
    return s

sl_rate  = mk_slider(0.665, 'spin rate', 1,   8,  3,        step=1)
sl_count = mk_slider(0.607, 'count',     2,  10,  6,        step=1)
sl_lam   = mk_slider(0.549, 'λ',         0.03, 0.25, LAMBDA_PDG, step=0.01, col='#303048')
sl_spf   = mk_slider(0.491, 'speed',     1,  12,  3,        step=1)

sl_rate.on_changed( lambda v: S.update(spin_rate=int(v)))
sl_count.on_changed(lambda v: S.update(count=int(v)))
sl_lam.on_changed(  lambda v: [S.update(lam=float(v)), refresh_info()])
sl_spf.on_changed(  lambda v: S.update(spf=int(v)))

# ── Action buttons ────────────────────────────────────────────────────────────
_aw = 0.365;  _ah = 0.052
b_pause   = mkbtn([_sx, 0.420, _aw, _ah], 'PAUSE')
b_clear   = mkbtn([_sx, 0.358, _aw, _ah], 'CLEAR')
b_reflect = mkbtn([_sx, 0.296, _aw, _ah], 'REFLECT: OFF', 8)
b_quit    = mkbtn([_sx, 0.234, _aw, _ah], 'QUIT')

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

# ─── Drag & drop ─────────────────────────────────────────────────────────────
GRAB_R  = 6
DRAG_T  = 6

drag = dict(on=False, oci=0, ocj=0, charge=0, ci=0, cj=0,
            sx=0.0, sy=0.0, moving=False)

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
    ci, cj = g
    found = find_vortex(ci, cj, GRAB_R)
    if found:
        ch = winding(found[0], found[1])
        if ch != 0:
            drag.update(on=True, oci=found[0], ocj=found[1], charge=ch,
                        ci=found[0], cj=found[1], sx=event.x, sy=event.y, moving=False)
            return
    do_place(ci, cj)

def on_move(event):
    if not drag['on'] or event.inaxes is not ax:
        return
    dx = event.x - drag['sx'];  dy = event.y - drag['sy']
    if not drag['moving'] and (dx*dx+dy*dy)**0.5 > DRAG_T:
        drag['moving'] = True
    if drag['moving']:
        g = eg(event)
        if g:
            drag['ci'] = max(6, min(N-7, g[0]))
            drag['cj'] = max(6, min(N-7, g[1]))

def on_release(event):
    if not drag['on']:
        return
    if drag['moving']:
        oci, ocj = drag['oci'], drag['ocj']
        nci, ncj = drag['ci'],  drag['cj']
        ch = drag['charge']
        if (nci-CY)**2+(ncj-CX)**2 < (R_ACTIVE-6)**2 and (oci!=nci or ocj!=ncj):
            remove_vortex(oci, ocj, ch)
            place_vortex(nci, ncj, ch)
    drag['on'] = False;  drag['moving'] = False

def on_key(event):
    k = event.key
    if k == ' ':            on_pause()
    elif k in ('r', 'R'):   do_reset()
    elif k in ('q','Q','escape'): plt.close('all'); sys.exit(0)

fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('motion_notify_event',  on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event',      on_key)

# ─── Animation ────────────────────────────────────────────────────────────────
def update(frame):
    if not S['paused']:
        for _ in range(S['spf']):
            step()
    img_h.set_array(to_rgb())
    if drag['on'] and drag['moving']:
        drag_circ.set_center((drag['cj'], drag['ci']));  drag_circ.set_alpha(0.9)
        drag_lbl.set_position((drag['cj'], drag['ci']))
        drag_lbl.set_text('⊕' if drag['charge'] > 0 else '⊖');  drag_lbl.set_alpha(0.95)
        drag_ln.set_data([drag['ocj'], drag['cj']], [drag['oci'], drag['ci']])
    else:
        drag_circ.set_alpha(0);  drag_lbl.set_alpha(0);  drag_ln.set_data([],[])
    return [img_h, drag_circ, drag_lbl, drag_ln]

ani = animation.FuncAnimation(fig, update, interval=25, blit=True, cache_frame_data=False)

print("Abelian-Higgs Vortex Simulation v3")
print(f"PDG 2024: λ={LAMBDA_PDG}  sin²θW={SIN2_TW}  gZ={G_Z:.3f}  eW={E_W:.3f}")
print(f"Grid {N}×{N}  active radius {R_ACTIVE}  steps/frame 1–12")
print()
print("  Click sim           place selected block")
print("  Drag vortex core    move vortex")
print("  SPACE / button      pause / resume")
print("  R                   reset field")
print("  Q / Escape          quit")
plt.tight_layout(pad=0)
plt.show()
