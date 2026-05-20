# Abelian-Higgs Vortex Simulation

An interactive 2D physics sandbox simulating topological vortices in the Abelian-Higgs model, using constants from the Particle Data Group (PDG) 2025 Review of Particle Physics.

![Python](https://img.shields.io/badge/python-3.8%2B-blue) ![numpy](https://img.shields.io/badge/numpy-required-orange) ![matplotlib](https://img.shields.io/badge/matplotlib-required-orange)

---

## What it simulates

The simulation evolves a complex scalar field φ = p1 + ip2 governed by the damped Abelian-Higgs equations on a 256×256 grid. Topological vortices (quantised phase singularities) are placed interactively and evolve under the field dynamics. The physics is analogous to vortices in a Type II superconductor or cosmic strings in the early universe.

**PDG 2025 constants used:**

| Quantity | Value | Source |
|---|---|---|
| Higgs mass m_H | 125.20 ± 0.11 GeV | PDG 2025 |
| Higgs VEV v | 246.22 GeV | PDG 2025 |
| sin²θ_W (MS-bar at M_Z) | 0.23122 | PDG 2025 |
| λ = m²_H / (2v²) | 0.13 | Derived |

---

## Requirements

```bash
pip install numpy matplotlib
```

---

## Running

```bash
python vortex_sim_v4-17a.py
```

---

## Controls

| Action | Control |
|---|---|
| Place vortex (+1) | Left-click in simulation area |
| Place antivortex (−1) | Right-click anywhere in simulation area |
| Undo placement | Ctrl+Z |
| Pause / Resume | Space or PAUSE button |
| Reset field | R or CLEAR button |
| Save state | File menu → Save state… (Ctrl+S) |
| Load state | File menu → Load state… (Ctrl+O) |
| Quit | Q / Escape / QUIT button |

---

## Features

### Vortex types
Place single vortices (+1 or −1 charge), pairs, twins, strings, and rings via the block selector buttons.

### Spin
Give vortices an initial velocity kick (CW or CCW) and a continuous phase rotation via the gauge field A₀.

### View modes (View menu)
| Mode | Description |
|---|---|
| Phase | HSV-like: colour = arg φ, brightness = \|φ\| |
| Energy | Field energy density — black → red → yellow → white |
| Topo | Topological charge density — blue = +1, red = −1 |

### Overlays (View menu)
- **Vortex core markers** — cyan dots on +1 vortices, orange on −1, detected via amplitude minima and topological charge sign
- **Phase gradient arrows** — white quiver arrows showing the probability current J = p1∇p2 − p2∇p1
- **Smooth interpolation** — bilinear rendering

### Info overlay (View menu — all off by default)
Each item is individually toggleable with zero compute cost when off:
- Placement count
- Topological charge ∫q dA (true winding number, measured live)
- Regime indicator — κ = √(2λ)/g_W and Type I/II label (updates live as λ slider moves)
- λ, sin²θ_W, e_W
- Mode / pause status

### Physics controls
| Slider | Effect |
|---|---|
| λ | Higgs self-coupling — controls vortex core size |
| speed | Steps per frame (1–16) |
| damp | 1.000 = Hamiltonian, 0.999 = light dissipation, 0.990 = heavy damping |
| spin rate | Angular velocity of spin sources |
| count | Number of vortices in string / ring patterns |

### Save / load state (File menu)
Save the full field state (φ, φ', λ, spin sources) to a `.npz` file and reload it in any future session. The λ slider resyncs automatically on load.

### Boundary modes
- **Absorbing (default)** — PML (Perfectly Matched Layer) damps outgoing radiation
- **Reflective** — hard circular wall; vortices bounce back

---

## Version history

| Version | Key changes |
|---|---|
| v4.17a | Save/load state (File menu), right-click antivortex, energy flicker fix, physics time budget |
| v4.17 | Performance pass: pre-allocated scratch arrays, slicing Laplacian, 30 ms interval, smooth interpolation default |
| v4.16 | Type I/II regime indicator (κ readout), λ slider extended to [0.005, 0.30] |
| v4.15 | Soft edge boundary (smoothstep fade, zero physics impact) |
| v4.15b | Clip path boundary (vector circle + soft edge combined) |
| v4.14 | Toggleable info overlay, topo charge on/off, UNDO button removed |
| v4.13 | View menu (Qt), vortex core markers, phase gradient arrows, smooth interpolation |
| v4.12 | PDG 2025 constants (sin²θ_W = 0.23122), dynamic info bar |
| v4.11 | Display mode toggle (phase / energy / topo) |
| v4.2 | Damping slider, 5-point numpy-roll Laplacian |

---

## Physics notes

- **Integrator**: damped leapfrog — preserves vacuum exactly; `damp=1.000` gives Hamiltonian dynamics
- **Laplacian**: 5-point stencil via slice views (no scipy, no array copies on interior)
- **Topological charge**: ∫q dA = ∫(∂ₓn̂ × ∂_yn̂)/2π dA, integrated over the active field; always rounds to an integer
- **PML**: cubic ramp absorbing layer from r=0.36N to r=0.48N suppresses boundary reflections
- **Vortex core size** scales as ξ ∝ 1/√(2λ); at λ=0.13, ξ ≈ 2 grid cells
- **Ginzburg-Landau parameter** κ = √(2λ)/g_W; Type I κ < 1/√2, Type II κ > 1/√2; crossover at λ ≈ 0.0225

---

## Citation

S. Navas et al. (Particle Data Group), Phys. Rev. D 110, 030001 (2024) and 2025 update.
