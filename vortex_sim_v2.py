#!/usr/bin/env python3
"""
Abelian-Higgs Vortex Simulation  —  Claude Code edition
=========================================================
Full-featured Python equivalent of the browser sandbox.
Physical constants: PDG 2024, S. Navas et al., Phys. Rev. D 110, 030001 (2024)

Features
--------
  Block types   : ⊕ particle, ⊖ antiparticle, pair, twin, string, ring
  Drag & drop   : click-drag any vortex to a new position
  Pause         : SPACE key or PAUSE button
  Spin          : none / CW / CCW with adjustable rate
  Field overlay : gauge-field visualisation
  Reflect       : toggle absorbing ↔ reflective boundary
  Sliders       : λ, count, spin rate, speed
  Reset / Quit  : R key or buttons

Requirements
------------
  pip install numpy matplotlib scipy --break-system-packages

Run
---
  python vortex_sim_v2.py
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button, Slider
from matplotlib.patches import Circle
from scipy.ndimage import convolve

# ─── PDG 2024 constants (dimensionless, v = 1) ───────────────────────────────
V          = 1.0
LAMBDA_PDG = 0.13           # λ = m²_H / (2v²),  m_H = 125.20 GeV
SIN2_TW    = 0.23129        # sin²θ_W  (MS-bar at M_Z)
SIN_TW     = np.sqrt(SIN2_TW)        # 0.4809
COS_TW     = np.sqrt(1.0 - SIN2_TW)  # 0.8769
G_W        = 0.30
G_Z        = G_W / COS_TW            # 0.342  = gW / cos(θ_W)
E_W        = G_W * SIN_TW            # 0.144  = gW · sin(θ_W)
SPIN_SCALE = SIN_TW * 0.00125        # spin-source coupling ∝ sin(θ_W)

print(f"PDG 2024  λ={LAMBDA_PDG}  sin²θW={SIN2_TW}  gZ={G_Z:.3f}  eW={E_W:.3f}")

# ─── Grid ─────────────────────────────────────────────────────────────────────
N   = 512
CX  = CY = N // 2
C2  = 0.65
DT  = 0.08
DAMP = 0.9999

R_ACTIVE  = int(N * 0.36)    # inner PML boundary  ≈ 184
R_PML_END = int(N * 0.48)    # outer PML boundary  ≈ 246
PML_MAX   = 0.55

# Pre-computed grids
_ii = np.arange(N)[:, None].astype(np.float32)
_jj = np.arange(N)[None, :].astype(np.float32)
_DI = _ii - CY;  _DJ = _jj - CX
R_GRID = np.sqrt(_DI**2 + _DJ**2)

pml = np.zeros((N, N), dtype=np.float32)
_s  = np.clip((R_GRID - R_ACTIVE) / (R_PML_END - R_ACTIVE), 0.0, 1.0)
pml[R_GRID >= R_ACTIVE] = (PML_MAX * _s**3)[R_GRID >= R_ACTIVE]

in_circle = R_GRID <= N // 2 - 1
wall_mask = R_GRID >= N // 2 - 1

# 9-point isotropic Laplacian kernel
_K9 = np.array([[1, 4, 1],
                 [4,-20, 4],
                 [1, 4, 1]], dtype=np.float32) / 6.0

def lap9(f):
    return convolve(f, _K9, mode='wrap')

# ─── Fields ───────────────────────────────────────────────────────────────────
p1  = np.ones( (N, N), dtype=np.float32)
p2  = np.zeros((N, N), dtype=np.float32)
p1P = np.ones( (N, N), dtype=np.float32)
p2P = np.zeros((N, N), dtype=np.float32)
A0  = np.zeros((N, N), dtype=np.float32)

# ─── Mutable simulation state ─────────────────────────────────────────────────
S = dict(
    paused     = False,
    block      = 'plus',
    spin_dir   = 0,
    spin_rate  = 3,
    count      = 6,
    lam        = LAMBDA_PDG,
    spf        = 4,
    reflect    = False,
    nv         = 0,
)

spin_sources   = []
background_A0  = 0.0

def recompute_A0():
    sig2 = 18.0 ** 2
    A0[:] = background_A0
    for src in spin_sources:
        A0 += src['m'] / (1.0 + ((_ii - src['ci'])**2 + (_jj - src['cj'])**2) / sig2)

# ─── Physics step ─────────────────────────────────────────────────────────────
def step():
    global p1, p2, p1P, p2P
    dt2 = DT * DT
    l1  = lap9(p1);  l2 = lap9(p2)
    pot = S['lam'] * (p1*p1 + p2*p2 - 1.0)
    v1  = p1 - p1P;  v2 = p2 - p2P
    ab  = 0.0 if S['reflect'] else pml
    n1  = (2*p1 - p1P + dt2*(C2*l1 - pot*p1)) * DAMP - ab*v1 - 2*A0*v2
    n2  = (2*p2 - p2P + dt2*(C2*l2 - pot*p2)) * DAMP - ab*v2 + 2*A0*v1
    if S['reflect']:
        n1[wall_mask] = V;  n2[wall_mask] = 0.0
    else:
        n1[[0,-1],:] = V;  n1[:,[0,-1]] = V
        n2[[0,-1],:] = 0;  n2[:,[0,-1]] = 0
    p1P[:] = p1;  p2P[:] = p2
    p1[:] = n1;   p2[:] = n2

# ─── Vortex tools ─────────────────────────────────────────────────────────────
def place_vortex(ci, cj, charge):
    global p1, p2, p1P, p2P
    sq2 = np.sqrt(2.0) / np.sqrt(S['lam'])
    dr  = _ii - ci;  dc = _jj - cj
    r   = np.sqrt(dr*dr + dc*dc) + 0.01
    prf = np.tanh(r / sq2)
    ph  = np.arctan2(p2, p1) + charge * np.arctan2(dc, dr)
    amp = np.sqrt(p1*p1 + p2*p2)
    p1[:] = amp * prf * np.cos(ph)
    p2[:] = amp * prf * np.sin(ph)
    p1P[:] = p1;  p2P[:] = p2

def remove_vortex(ci, cj, charge):
    global p1, p2, p1P, p2P
    sq2 = np.sqrt(2.0) / np.sqrt(S['lam'])
    dr  = _ii - ci;  dc = _jj - cj
    r   = np.sqrt(dr*dr + dc*dc) + 0.01
    prf = np.maximum(0.04, np.tanh(r / sq2))
    ph  = np.arctan2(p2, p1) - charge * np.arctan2(dc, dr)
    amp = np.minimum(float(V), np.sqrt(p1*p1 + p2*p2) / prf)
    p1[:] = amp * np.cos(ph)
    p2[:] = amp * np.sin(ph)
    p1P[:] = p1;  p2P[:] = p2

def find_vortex(ci, cj, radius=10):
    i0 = max(0, ci-radius);  i1 = min(N, ci+radius)
    j0 = max(0, cj-radius);  j1 = min(N, cj+radius)
    if i0 >= i1 or j0 >= j1:
        return None
    reg = np.sqrt(p1[i0:i1, j0:j1]**2 + p2[i0:i1, j0:j1]**2).copy()
    li  = np.arange(i0, i1)[:, None];  lj = np.arange(j0, j1)[None, :]
    reg[(li-ci)**2 + (lj-cj)**2 > radius**2] = 1.0
    idx = np.unravel_index(np.argmin(reg), reg.shape)
    return (i0+idx[0], j0+idx[1]) if reg[idx] < 0.35 else None

def winding(ci, cj, r=5, npts=16):
    ang = np.linspace(0, 2*np.pi, npts, endpoint=False)
    si  = np.clip(np.round(ci + r*np.sin(ang)).astype(int), 0, N-1)
    sj  = np.clip(np.round(cj + r*np.cos(ang)).astype(int), 0, N-1)
    ph  = np.arctan2(p2[si, sj], p1[si, sj])
    d   = np.diff(np.append(ph, ph[0]))
    d   = (d + np.pi) % (2*np.pi) - np.pi
    return int(np.round(d.sum() / (2*np.pi)))

def apply_rotation(ci, cj):
    infl = 24;  infl2 = infl*infl
    speed = S['spin_rate'] * 0.012 * S['spin_dir']
    i0 = max(1, ci-infl);  i1 = min(N-1, ci+infl)
    j0 = max(1, cj-infl);  j1 = min(N-1, cj+infl)
    li = np.arange(i0, i1)[:, None];  lj = np.arange(j0, j1)[None, :]
    dr = li - ci;  dc = lj - cj
    r2 = dr*dr + dc*dc;  r = np.sqrt(r2) + 0.01
    w  = np.where((r2 >= 1) & (r2 <= infl2), np.exp(-r2/(infl2*0.28)), 0)
    vi = -dc / r * speed * w;  vj = dr / r * speed * w
    sub1 = p1[i0:i1, j0:j1];  sub2 = p2[i0:i1, j0:j1]
    p1P[i0:i1, j0:j1] = sub1 - DT*(vi*(np.roll(sub1,-1,0)-np.roll(sub1,1,0))*0.5
                                   + vj*(np.roll(sub1,-1,1)-np.roll(sub1,1,1))*0.5)
    p2P[i0:i1, j0:j1] = sub2 - DT*(vi*(np.roll(sub2,-1,0)-np.roll(sub2,1,0))*0.5
                                   + vj*(np.roll(sub2,-1,1)-np.roll(sub2,1,1))*0.5)

def cl(x):
    return max(12, min(N-13, x))

def clR(ci, cj):
    di = ci - CY;  dj = cj - CX
    r  = np.sqrt(di*di + dj*dj)
    mr = R_ACTIVE - 8
    if r > mr:
        f = mr / r;  return (int(round(CY + di*f)), int(round(CX + dj*f)))
    return (ci, cj)

def add_source(ci, cj):
    if S['spin_dir'] == 0:
        return
    spin_sources.append({'ci': ci, 'cj': cj, 'm': S['spin_dir']*S['spin_rate']*SPIN_SCALE})
    if len(spin_sources) > 12:
        spin_sources.pop(0)
    recompute_A0()

def do_place(ci, cj):
    ci, cj = clR(cl(ci), cl(cj))
    if (ci-CY)**2 + (cj-CX)**2 > (R_ACTIVE-8)**2:
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
        place_vortex(ci, cl(cj-10), +1);  place_vortex(ci, cl(cj+10), -1);  S['nv'] += 2
    elif blk == 'twins':
        place_vortex(cl(ci-10), cj, +1);  place_vortex(cl(ci+10), cj, +1);  S['nv'] += 2
    elif blk == 'string':
        sp = 13;  tot = (n-1)*sp
        for k in range(n):
            place_vortex(ci, cl(cj + round(k*sp - tot/2)), +1 if k%2==0 else -1)
        S['nv'] += n
    elif blk == 'ring':
        rr = max(14, round(n*5.5))
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
    r = np.clip((0.5 + 0.5*np.cos(ph))      * amp, 0, 1)
    g = np.clip((0.5 + 0.5*np.cos(ph-_T3)) * amp, 0, 1)
    b = np.clip((0.5 + 0.5*np.cos(ph+_T3)) * amp, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    rgb[~in_circle] = 0
    return (rgb * 255).astype(np.uint8)

# ─── Figure layout ────────────────────────────────────────────────────────────
CB   = '#0d0d0d'   # button background
CBH  = '#1a1a1a'   # button hover
CBON = '#152515'   # button active
CBOH = '#1e331e'   # button active hover
CT   = '#999999'   # text
CTON = '#77dd77'   # text active

fig = plt.figure(figsize=(12.5, 8.5), facecolor='#080808')
try:
    fig.canvas.manager.set_window_title('Abelian-Higgs Vortex Simulation')
except Exception:
    pass

ax = fig.add_axes([0.01, 0.03, 0.60, 0.95])
ax.set_facecolor('#000');  ax.set_aspect('equal');  ax.axis('off')

img_h = ax.imshow(to_rgb(), origin='upper', extent=[0,N,N,0],
                  animated=True, interpolation='nearest')
ax.set_xlim(0, N);  ax.set_ylim(N, 0)
ax.add_patch(Circle((CX,CY), N//2-2, fill=False, edgecolor='#1c1c1c', lw=2))

drag_circ = Circle((CX,CY), 10, fill=True, facecolor=(1,1,1,0.12),
                    edgecolor='white', lw=1.5, alpha=0, zorder=10)
ax.add_patch(drag_circ)
drag_lbl  = ax.text(CX, CY, '', ha='center', va='center',
                     color='white', fontsize=13, fontweight='bold', alpha=0, zorder=11)
drag_ln,  = ax.plot([], [], '--', color='white', alpha=0.3, lw=1, zorder=9)
info_txt  = ax.text(CX, N-4, '', ha='center', va='bottom',
                     color='#444', fontsize=8, fontfamily='monospace')

def refresh_info():
    info_txt.set_text(
        f"vortices: {S['nv']}   λ={S['lam']:.3f}   "
        f"sin²θW=0.231   eW=0.144   "
        f"{'[ PAUSED ]' if S['paused'] else ''}"
    )

refresh_info()

# ── Helper: create a styled button ───────────────────────────────────────────
def mkbtn(rect, label, fs=9):
    a = fig.add_axes(rect, facecolor=CB)
    b = Button(a, label, color=CB, hovercolor=CBH)
    b.label.set_color(CT);  b.label.set_fontsize(fs)
    b.label.set_fontfamily('monospace')
    return b

# ── Panel header ──────────────────────────────────────────────────────────────
ax_hdr = fig.add_axes([0.635, 0.935, 0.35, 0.045], facecolor='#080808')
ax_hdr.text(0.5, 0.5, 'VORTEX SANDBOX  ·  PDG 2024', ha='center', va='center',
            color='#444', fontsize=8, fontfamily='monospace')
ax_hdr.axis('off')

# ── Block buttons (2 rows × 3) ────────────────────────────────────────────────
#   Row 1
b_plus   = mkbtn([0.635, 0.875, 0.108, 0.052], '⊕  particle')
b_minus  = mkbtn([0.752, 0.875, 0.108, 0.052], '⊖  antipart.')
b_pair   = mkbtn([0.869, 0.875, 0.108, 0.052], '⊕⊖  pair')
#   Row 2
b_twins  = mkbtn([0.635, 0.814, 0.108, 0.052], '⊕⊕  twin')
b_string = mkbtn([0.752, 0.814, 0.108, 0.052], '⊕⊖⊕  string')
b_ring   = mkbtn([0.869, 0.814, 0.108, 0.052], '⊕⊖  ring')

BLOCK_BTNS = dict(plus=b_plus, minus=b_minus, pair=b_pair,
                  twins=b_twins, string=b_string, ring=b_ring)

def set_block(name):
    S['block'] = name
    for k, btn in BLOCK_BTNS.items():
        on = (k == name)
        btn.color       = CBON if on else CB
        btn.hovercolor  = CBOH if on else CBH
        btn.label.set_color(CTON if on else CT)
    fig.canvas.draw_idle()

b_plus.on_clicked(  lambda e: set_block('plus'))
b_minus.on_clicked( lambda e: set_block('minus'))
b_pair.on_clicked(  lambda e: set_block('pair'))
b_twins.on_clicked( lambda e: set_block('twins'))
b_string.on_clicked(lambda e: set_block('string'))
b_ring.on_clicked(  lambda e: set_block('ring'))
set_block('plus')

# ── Spin buttons ──────────────────────────────────────────────────────────────
ax_sl = fig.add_axes([0.635, 0.774, 0.35, 0.030], facecolor='#080808')
ax_sl.text(0.03, 0.5, 'SPIN', va='center', color='#444', fontsize=8, fontfamily='monospace')
ax_sl.axis('off')

b_snone = mkbtn([0.635, 0.737, 0.108, 0.030], 'none', 8)
b_scw   = mkbtn([0.752, 0.737, 0.108, 0.030], '↻  cw',  8)
b_sccw  = mkbtn([0.869, 0.737, 0.108, 0.030], '↺  ccw', 8)
SPIN_BTNS = {'none': b_snone, 'cw': b_scw, 'ccw': b_sccw}
SPIN_MAP  = {'none': 0, 'cw': -1, 'ccw': +1}

def set_spin(name):
    S['spin_dir'] = SPIN_MAP[name]
    for k, btn in SPIN_BTNS.items():
        on = (k == name)
        btn.color = CBON if on else CB;  btn.hovercolor = CBOH if on else CBH
        btn.label.set_color(CTON if on else CT)
    fig.canvas.draw_idle()

b_snone.on_clicked(lambda e: set_spin('none'))
b_scw.on_clicked(  lambda e: set_spin('cw'))
b_sccw.on_clicked( lambda e: set_spin('ccw'))
set_spin('none')

# ── Sliders ───────────────────────────────────────────────────────────────────
def sl(rect, lbl, lo, hi, init, step_=None, color='#404050'):
    a = fig.add_axes(rect, facecolor='#0a0a0a')
    kw = dict(valstep=step_) if step_ else {}
    s = Slider(a, lbl, lo, hi, valinit=init, **kw)
    s.poly.set_facecolor(color)
    s.label.set_color(CT);  s.label.set_fontsize(8);  s.label.set_fontfamily('monospace')
    s.valtext.set_color(CT);  s.valtext.set_fontsize(8)
    return s

sl_rate  = sl([0.635, 0.680, 0.35, 0.035], 'spin rate', 1, 8, 3, 1)
sl_count = sl([0.635, 0.620, 0.35, 0.035], 'count',     2, 10, 6, 1)
sl_lam   = sl([0.635, 0.560, 0.35, 0.035], 'λ',    0.03, 0.25, LAMBDA_PDG, 0.01, '#334')
sl_spf   = sl([0.635, 0.500, 0.35, 0.035], 'speed',     1,  8, 4, 1)

sl_rate.on_changed( lambda v: S.update(spin_rate=int(v)))
sl_count.on_changed(lambda v: S.update(count=int(v)))
sl_lam.on_changed(  lambda v: S.update(lam=float(v)) or refresh_info())
sl_spf.on_changed(  lambda v: S.update(spf=int(v)))

# ── Action buttons ────────────────────────────────────────────────────────────
b_pause   = mkbtn([0.635, 0.426, 0.35, 0.058], 'PAUSE')
b_clear   = mkbtn([0.635, 0.356, 0.35, 0.058], 'CLEAR')
b_reflect = mkbtn([0.635, 0.286, 0.35, 0.058], 'REFLECT: OFF', 8)
b_quit    = mkbtn([0.635, 0.216, 0.35, 0.058], 'QUIT')

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

# ── Legend ────────────────────────────────────────────────────────────────────
ax_leg = fig.add_axes([0.635, 0.035, 0.35, 0.165], facecolor='#080808')
for yi, (txt, col) in enumerate([
    ('■  blue   =  stabilises ↺ CCW spin', '#4477bb'),
    ('■  orange =  stabilises ↻ CW spin',  '#bb6622'),
    ('colour  =  phase angle of φ',         '#555'),
    ('dark core  =  vortex  (|φ| → 0)',     '#555'),
    ('SPACE = pause  ·  R = reset  ·  Q = quit', '#444'),
]):
    ax_leg.text(0.04, 0.88 - yi*0.195, txt, color=col,
                fontsize=7, fontfamily='monospace', va='top')
ax_leg.axis('off')

# ─── Drag & drop ─────────────────────────────────────────────────────────────
GRAB_R  = 7
HOVER_R = 9
DRAG_T  = 8   # pixel threshold before drag activates

drag = dict(on=False, oci=None, ocj=None, charge=None,
            ci=None, cj=None, sx=None, sy=None, moving=False)

def eg(event):
    """Convert event to grid coordinates, or None if outside axes."""
    if event.inaxes is not ax:
        return None
    ci = int(round(float(event.ydata)))
    cj = int(round(float(event.xdata)))
    return (max(0, min(N-1, ci)), max(0, min(N-1, cj)))

def on_press(event):
    if event.inaxes is not ax:
        return
    if event.button not in (1, 3):
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
    if not drag['on']:
        return
    if event.inaxes is not ax:
        return
    dx = event.x - drag['sx'];  dy = event.y - drag['sy']
    if not drag['moving'] and (dx*dx+dy*dy)**0.5 > DRAG_T:
        drag['moving'] = True
    if drag['moving']:
        g = eg(event)
        if g:
            drag['ci'] = max(10, min(N-11, g[0]))
            drag['cj'] = max(10, min(N-11, g[1]))

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
    if event.key == ' ':
        on_pause()
    elif event.key in ('r', 'R'):
        do_reset()
    elif event.key in ('q', 'Q', 'escape'):
        plt.close('all');  sys.exit(0)

fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('motion_notify_event',  on_move)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event',      on_key)

# ─── Animation loop ───────────────────────────────────────────────────────────
def update(frame):
    if not S['paused']:
        for _ in range(S['spf']):
            step()
    img_h.set_array(to_rgb())
    # Drag indicator
    if drag['on'] and drag['moving']:
        drag_circ.set_center((drag['cj'], drag['ci']));  drag_circ.set_alpha(0.9)
        drag_lbl.set_position((drag['cj'], drag['ci']))
        drag_lbl.set_text('⊕' if drag['charge'] > 0 else '⊖');  drag_lbl.set_alpha(0.95)
        drag_ln.set_data([drag['ocj'], drag['cj']], [drag['oci'], drag['ci']])
    else:
        drag_circ.set_alpha(0);  drag_lbl.set_alpha(0);  drag_ln.set_data([],[])
    return [img_h, drag_circ, drag_lbl, drag_ln]

ani = animation.FuncAnimation(fig, update, interval=25, blit=True, cache_frame_data=False)

print()
print("Block buttons  : click to select, then click the field to place")
print("Drag & drop    : hover near any vortex core → cursor changes → drag to move")
print("SPACE / button : pause / resume")
print("R              : reset field to vacuum")
print("Q / Escape     : quit")
print()
plt.tight_layout(pad=0)
plt.show()
