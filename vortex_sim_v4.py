#!/usr/bin/env python3
"""
Abelian-Higgs Vortex Simulation  —  Claude Code edition  v4
=============================================================
Fixes vs v3:
  - Damping bug fixed: vacuum is now preserved exactly.
    v3 used  (2·p − p_prev + dt²F) · DAMP  which decayed even at rest.
    v4 uses  (1+DAMP)·p − DAMP·p_prev + dt²F  (proper damped leapfrog).

Physical constants verified against PDG 2024:
  m_H  = 125.20 ± 0.11 GeV   →  λ = m²_H/(2v²) = 0.13
  v    = 246.22 GeV          (Higgs VEV, normalised to v=1 in sim)
  sin²θ_W = 0.23129          (MS-bar at M_Z)
  gZ/gW   = 1/cos(θ_W) = 1.140
  eW/gW   = sin(θ_W)   = 0.481
Source: S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024).

Requirements:
    pip install numpy matplotlib scipy --break-system-packages

Run:
    python vortex_sim_v4.py
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

    # Damped leapfrog — proper form that preserves vacuum exactly.
    # Derived from   v_{n+1/2} = DAMP · v_{n-1/2} + dt · F
    # which gives    φ_{n+1} = (1+DAMP)·φ_n − DAMP·φ_{n-1} + dt²·F
    # For vacuum (φ=v, F=0):  (1+DAMP)·v − DAMP·v = v.  No drift.
    n1 = (1.0+DAMP)*p1 - DAMP*p1P + dt2*(C2*l1 - pot*p1) - ab*v1 - 2.0*A0*v2
    n2 = (1.0+DAMP)*p2 - DAMP*p2P + dt2*(C2*l2 - pot*p2) - ab*v2 + 2.0*A0*v1

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
    fig.canvas.manager.set_window_title('Abelian-Higgs Vortex Simulation  v4')
except Exception:
    pass

# Simulation axes
ax = fig.add_axes([0.01, 0.03, 0.60, 0.95])
ax.set_facecolor('#000');  ax.set_aspect('equal');  ax.axis('off')
img_h = ax.imshow(to_rgb(), origin='upper', extent=[0,N,N,0],
                  animated=True, interpolation='nearest')
ax.set_xlim(0, N);  ax.set_ylim(N, 0)
ax.add_patch(Circle((CX, CY), N//2-2, fill=False, edgecolor='#1a1a1a', lw=1.5))

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
    do_place(g[0], g[1])

def on_key(event):
    k = event.key
    if k == ' ':            on_pause()
    elif k in ('r', 'R'):   do_reset()
    elif k in ('q','Q','escape'): plt.close('all'); sys.exit(0)

fig.canvas.mpl_connect('button_press_event',   on_press)
fig.canvas.mpl_connect('key_press_event',      on_key)

# ─── Animation ────────────────────────────────────────────────────────────────
def update(frame):
    if not S['paused']:
        for _ in range(S['spf']):
            step()
    img_h.set_array(to_rgb())
    return [img_h]

ani = animation.FuncAnimation(fig, update, interval=25, blit=True, cache_frame_data=False)

print("Abelian-Higgs Vortex Simulation v4 (damping bug fixed)")
print(f"PDG 2024: λ={LAMBDA_PDG}  sin²θW={SIN2_TW}  gZ={G_Z:.3f}  eW={E_W:.3f}")
print(f"Grid {N}×{N}  active radius {R_ACTIVE}  steps/frame 1–12")
print()
print("  Click sim           place selected block")
print("  SPACE / button      pause / resume")
print("  R                   reset field")
print("  Q / Escape          quit")
plt.tight_layout(pad=0)
plt.show()
