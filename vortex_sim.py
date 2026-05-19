"""
Abelian-Higgs Vortex Field Simulation
======================================
Python/NumPy implementation for Claude Code.
Physical constants from PDG 2024: S. Navas et al., Phys. Rev. D 110, 030001 (2024)

Grid: 512×512  (vs 190×190 in browser — ~7× more resolution)
Uses NumPy vectorised operations — roughly 50× faster than the JS loop version.

Requirements:
    pip install numpy matplotlib scipy --break-system-packages

Optional (for video export):
    pip install imageio[ffmpeg] --break-system-packages

Run:
    python vortex_sim.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PDG 2024 Physical constants (dimensionless, normalised to v = 1)
# ─────────────────────────────────────────────────────────────────────────────
V          = 1.0          # Higgs VEV (normalised; physical: 246.22 GeV)
LAMBDA     = 0.13         # Higgs self-coupling: λ = m²_H/(2v²), m_H = 125.20 GeV
SIN2_TW    = 0.23129      # Weak mixing angle sin²θ_W (MS-bar at M_Z, PDG 2024)
SIN_TW     = np.sqrt(SIN2_TW)        # 0.4809
COS_TW     = np.sqrt(1.0 - SIN2_TW)  # 0.8769
G_W        = 0.30         # SU(2) W coupling (normalised reference)
G_Z        = G_W / COS_TW            # 0.342  gZ = gW/cos(θW)
E_W        = G_W * SIN_TW            # 0.144  e = gW·sin(θW)

print(f"PDG 2024 values loaded:")
print(f"  λ  = {LAMBDA}   (Higgs self-coupling)")
print(f"  sin²θW = {SIN2_TW}  (weak mixing angle)")
print(f"  gW = {G_W:.3f}  gZ = {G_Z:.3f}  eW = {E_W:.3f}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# Grid and simulation parameters
# ─────────────────────────────────────────────────────────────────────────────
N          = 512          # Grid size — 512×512 vs 190×190 in browser
CX, CY     = N // 2, N // 2
C2         = 0.65         # Wave speed squared
DT         = 0.08         # Time step (conservative for N=512)
DAMP       = 0.9999       # Field damping per step
SPF        = 4            # Steps per rendered frame
lambda_sim = LAMBDA       # Mutable during runtime

# Circular PML absorbing boundary
R_ACTIVE   = int(N * 0.36)   # ~184 — active region radius
R_PML_END  = int(N * 0.48)   # ~246 — PML outer radius
PML_MAX    = 0.55

# ─────────────────────────────────────────────────────────────────────────────
# Precomputed masks
# ─────────────────────────────────────────────────────────────────────────────
i_idx = np.arange(N)[:, None]
j_idx = np.arange(N)[None, :]
di = i_idx - CY
dj = j_idx - CX
r_grid = np.sqrt(di**2 + dj**2)

# Circular PML — velocity-damping, absorbs outgoing waves
pml = np.zeros((N, N), dtype=np.float32)
mask_pml = r_grid >= R_ACTIVE
s_pml = np.clip((r_grid - R_ACTIVE) / (R_PML_END - R_ACTIVE), 0, 1)
pml[mask_pml] = PML_MAX * s_pml[mask_pml] ** 3

# Display mask
in_circle = (r_grid <= N // 2 - 1).astype(bool)

# ─────────────────────────────────────────────────────────────────────────────
# Field arrays — complex scalar field φ = p1 + i·p2
# ─────────────────────────────────────────────────────────────────────────────
p1  = np.ones( (N, N), dtype=np.float32)   # Re(φ), initialised to vacuum
p2  = np.zeros((N, N), dtype=np.float32)   # Im(φ)
p1P = np.ones( (N, N), dtype=np.float32)   # Re(φ) at previous step
p2P = np.zeros((N, N), dtype=np.float32)   # Im(φ) at previous step

# Gauge field A₀ — local phase rotation rate from spinning sources
A0  = np.zeros((N, N), dtype=np.float32)

# ─────────────────────────────────────────────────────────────────────────────
# 9-point isotropic Laplacian kernel (eliminates square-grid anisotropy)
# Stencil: (4·axis + 1·diagonal − 20·centre) / 6
# ─────────────────────────────────────────────────────────────────────────────
def laplacian9(f):
    """9-point isotropic Laplacian using roll (periodic wrap, corrected by PML)."""
    axis = (np.roll(f, 1, 0) + np.roll(f, -1, 0) +
            np.roll(f, 1, 1) + np.roll(f, -1, 1))
    diag = (np.roll(np.roll(f,  1, 0),  1, 1) + np.roll(np.roll(f, -1, 0),  1, 1) +
            np.roll(np.roll(f,  1, 0), -1, 1) + np.roll(np.roll(f, -1, 0), -1, 1))
    return (4 * axis + diag - 20 * f) / 6.0

# ─────────────────────────────────────────────────────────────────────────────
# Simulation step
# ─────────────────────────────────────────────────────────────────────────────
def step():
    global p1, p2, p1P, p2P
    dt2 = DT ** 2

    l1 = laplacian9(p1)
    l2 = laplacian9(p2)

    # Mexican hat potential: V = λ(|φ|² − v²)²
    pot = lambda_sim * (p1**2 + p2**2 - 1.0)

    vel1 = p1 - p1P
    vel2 = p2 - p2P

    p1N = (2*p1 - p1P + dt2*(C2*l1 - pot*p1)) * DAMP - pml * vel1
    p2N = (2*p2 - p2P + dt2*(C2*l2 - pot*p2)) * DAMP - pml * vel2

    # Minimal U(1) gauge coupling — (D_t)²φ, D_t = ∂_t − iA₀
    # Correction: d²p1/dt² += −2A₀(dp2/dt), d²p2/dt² += +2A₀(dp1/dt)
    p1N -= 2.0 * A0 * vel2
    p2N += 2.0 * A0 * vel1

    # Hard vacuum boundary at outermost ring (backstop behind PML)
    p1N[[0, -1], :] = V;  p1N[:, [0, -1]] = V
    p2N[[0, -1], :] = 0;  p2N[:, [0, -1]] = 0

    p1P, p2P = p1.copy(), p2.copy()
    p1, p2   = p1N, p2N

# ─────────────────────────────────────────────────────────────────────────────
# Vortex placement
# ─────────────────────────────────────────────────────────────────────────────
def place_vortex(ci, cj, charge):
    """Stamp a vortex of given charge at grid position (ci, cj)."""
    global p1, p2, p1P, p2P
    sq2xi = np.sqrt(2.0) / np.sqrt(lambda_sim)
    dr = i_idx - ci
    dc = j_idx - cj
    r  = np.sqrt(dr**2 + dc**2) + 0.01
    profile = np.tanh(r / sq2xi)
    theta   = np.arctan2(dc, dr)
    amp     = np.sqrt(p1**2 + p2**2)
    phase   = np.arctan2(p2, p1) + charge * theta
    p1 = amp * profile * np.cos(phase)
    p2 = amp * profile * np.sin(phase)
    p1P = p1.copy()
    p2P = p2.copy()

def find_vortex(ci, cj, radius=10):
    """Find nearest vortex core (min |φ|) within radius grid cells."""
    i0, i1 = max(0, ci-radius), min(N, ci+radius)
    j0, j1 = max(0, cj-radius), min(N, cj+radius)
    region_amp = np.sqrt(p1[i0:i1, j0:j1]**2 + p2[i0:i1, j0:j1]**2)
    dist = np.sqrt((np.arange(i0,i1)[:,None]-ci)**2 + (np.arange(j0,j1)[None,:]-cj)**2)
    region_amp[dist > radius] = 1.0  # mask outside circle
    idx = np.unravel_index(np.argmin(region_amp), region_amp.shape)
    min_amp = region_amp[idx]
    if min_amp < 0.35:
        return (i0 + idx[0], j0 + idx[1])
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Colour mapping — phase → RGB
# ─────────────────────────────────────────────────────────────────────────────
T3 = 2 * np.pi / 3

def field_to_rgb():
    """Convert complex field to RGB image array (H×W×3, uint8)."""
    amp   = np.clip(np.sqrt(p1**2 + p2**2), 0, 1)
    phase = np.arctan2(p2, p1)
    r = np.clip((0.5 + 0.5 * np.cos(phase))     * amp, 0, 1)
    g = np.clip((0.5 + 0.5 * np.cos(phase - T3)) * amp, 0, 1)
    b = np.clip((0.5 + 0.5 * np.cos(phase + T3)) * amp, 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    # Mask outside circle to black
    rgb[~in_circle] = 0
    return (rgb * 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# Interactive Matplotlib display
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 7))
fig.patch.set_facecolor('#000')
ax.set_facecolor('#000')
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Abelian-Higgs Vortex Field  |  λ=0.13  sin²θW=0.231  eW=0.144',
             color='#aaa', fontsize=9, fontfamily='monospace', pad=8)

# Initial render
img_display = ax.imshow(field_to_rgb(), origin='upper',
                        extent=[0, N, N, 0], animated=True)

# Circle overlay
circle_patch = Circle((CX, CY), N//2 - 2, fill=False,
                       edgecolor='#222', linewidth=1.5)
ax.add_patch(circle_patch)

# Instructions text
info_text = ax.text(CX, N + 14, 'click to place vortex  |  shift+click = anti-vortex  |  q = quit',
                    color='#555', fontsize=8, ha='center', va='top',
                    fontfamily='monospace', transform=ax.transData)
ax.set_xlim(-5, N + 5)
ax.set_ylim(N + 20, -5)

click_charge = [1]  # mutable state for current charge (+1 or -1)

def on_click(event):
    if event.inaxes != ax:
        return
    ci, cj = int(round(event.ydata)), int(round(event.xdata))
    ci = max(10, min(N-11, ci))
    cj = max(10, min(N-11, cj))
    dist_center = np.sqrt((ci-CY)**2 + (cj-CX)**2)
    if dist_center > R_ACTIVE - 8:
        return
    charge = -1 if event.key == 'shift' else 1
    place_vortex(ci, cj, charge)
    print(f"Placed {'⊕' if charge>0 else '⊖'} vortex at ({ci}, {cj})")

def on_key(event):
    if event.key == 'q':
        print("Exiting.")
        plt.close()
        sys.exit(0)
    elif event.key == 'r':
        global p1, p2, p1P, p2P
        p1[:] = V; p2[:] = 0; p1P[:] = V; p2P[:] = 0
        print("Field reset to vacuum.")
    elif event.key == 'p':
        place_vortex(CY, CX - 20, +1)
        place_vortex(CY, CX + 20, -1)
        print("Placed ⊕⊖ pair at centre.")
    elif event.key == 't':
        place_vortex(CY, CX - 20, +1)
        place_vortex(CY, CX + 20, +1)
        print("Placed ⊕⊕ twin pair.")

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

frame_count = [0]

def update(frame):
    for _ in range(SPF):
        step()
    frame_count[0] += 1
    img_display.set_array(field_to_rgb())
    return [img_display]

ani = animation.FuncAnimation(
    fig, update,
    interval=30,      # ms between frames (~33fps target)
    blit=True,
    cache_frame_data=False
)

print("Controls:")
print("  Click          — place +1 vortex")
print("  Shift+click    — place −1 anti-vortex")
print("  p              — place ⊕⊖ annihilating pair at centre")
print("  t              — place ⊕⊕ repelling twin pair")
print("  r              — reset field to vacuum")
print("  q              — quit")
print()
print("Close the window or press q to exit.")
print()

plt.tight_layout()
plt.show()
