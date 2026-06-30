# Module `time_relaxation`

This IGM module performs **data assimilation by forward time relaxation**: it runs an iceflow forward simulation in which one or more state fields (typically the surface mass balance, basal sliding coefficient, bedrock topography, or ice thickness) are nudged at each time step so that the modelled state converges toward an observation. The whole inner loop runs inside `initialize()`; once it returns, the outer IGM loop is signalled to exit, so `time_relaxation` *replaces* the usual `time` module rather than running alongside it.

The driver is **fully generic**: there are no named modes. A relaxation run is described as a list of independent **steps**, each step being an orthogonal triple `(residual, update law, control)`. The same driver, with different parameters, reproduces a series of relaxation methods commonly used in glacier and ice-sheet modelling — see the catalogue in *Recovering classical methods* below.

When the apparent-mass-balance variant is used, `state.amb = state.smb − state.dhdt_obs` is computed inside `time_relaxation` itself (`_ensure_derived`, every iteration). All that is required is an SMB module (e.g. `smb_simple`) in `pre_processes` so `state.smb` is fresh, and a `dhdt` field in the input NetCDF (snapshotted to `state.dhdt_obs` at startup).

**Contributors:** G. Jouvet with help from Claude, based on original implementation by T. Frank.

---

The forward dynamics IGM solves at every iteration is the standard mass-conservation equation for ice thickness $h$:

$$\frac{\partial h}{\partial t} \;+\; \nabla\cdot(\bar{\mathbf u}\,h) \;=\; \mathrm{SMB},$$

where $\bar{\mathbf u}$ is the depth-averaged velocity (from the `iceflow` solver), $\mathrm{SMB}$ is the surface mass balance, and $\nabla\cdot(\bar{\mathbf u} h)$ is the flux divergence (`divflux`). The basal sliding law links the velocity to a sliding coefficient $C$ (`slidingco`) and the bed shear stress.

**Time-relaxation data assimilation** modifies this forward simulation by *nudging* one or more **control fields** $C \in \{\mathrm{SMB},\, h,\, z_b,\, z_s,\, C_{\text{slid}}, \dots\}$ so that some **residual** $r$ between modelled and observed quantities is driven toward zero:

$$r = r(\text{state}, \text{obs}) \quad \xrightarrow{\;\;\text{nudge}\;\;}\quad 0.$$

The nudge is a small per-iteration perturbation of $C$:

$$\boxed{\;C^{n+1} = \Phi\bigl(C^n,\; r,\; \alpha,\; \Delta t\bigr)\;}$$

with four built-in choices for $\Phi$ (the **update law**):

$$\begin{aligned}
\text{additive:}              \quad & C \leftarrow C + \alpha\, r\, \Delta t \\
\text{multiplicative:}        \quad & C \leftarrow C \exp(\alpha\, r\, \Delta t) \quad\text{(exact ODE integral)} \\
\text{multiplicative\_linear:}\quad & C \leftarrow C \,(1 + \alpha\, r\, \Delta t) \quad\text{(linearised)} \\
\text{replace:}               \quad & C \leftarrow \alpha\, r \quad\text{(absolute write; legacy "$C := f(\text{obs})$")}
\end{aligned}$$

and three built-in choices for the residual:

$$\begin{aligned}
\text{linear:}     \quad & r = T - M \\
\text{relative:}   \quad & r = (T - M) / \max(|T|, \varepsilon) \\
\text{log\_ratio:} \quad & r = \log\!\bigl(\max(M,\varepsilon)/\max(T,\varepsilon)\bigr)
\end{aligned}$$

where $T$ is the *target* state field (e.g. `usurf_obs`, `velsurf_magobs`, `amb`) and $M$ is the *current* state field (e.g. `usurf`, `velsurf_mag`, `divflux`).

### What can the control $C$ be?

The control field is *anything* on `state` that the forward model reads. Below are the same template `(residual, update, control)` filled in for five common methods. The driver itself is identical for all of them; only the YAML changes.

| method | control $C$ | residual $r$ | update law |
|---|---|---|---|
| **PISM force-to-thickness**                                     | $\mathrm{SMB}$ | $r = h^{\text{target}} - h$ (linear) | `replace`: $C \leftarrow \alpha\, r$ |
| **Apparent-mass-balance bed inversion** (Frank & van Pelt 2025) | $h$ + $z_s$ (two coupled steps) | $r = \mathrm{amb} - \nabla\!\cdot\!(\bar{\mathbf u} h)$ (linear, shared) | `additive`: $C \leftarrow C + \alpha\, r\, \Delta t$ |
| **Linear-multiplicative friction** (legacy IGM additive)        | $C_{\text{slid}}$ | $r = (\lvert\mathbf u_s^{\text{obs}}\rvert - \lvert\mathbf u_s\rvert)/\lvert\mathbf u_s^{\text{obs}}\rvert$ (relative) | `multiplicative_linear`: $C \leftarrow C\,(1 + \alpha\, r)$ with $\alpha=-1$ |

Several steps can run together in the same time loop — for instance a Frank/van Pelt bed inversion (two steps writing $h$ and $z_s$) plus a friction-inversion step writing $C_{\text{slid}}$, all sharing the time clock and the same `iceflow` solve. Two steps can share a residual via `shares_residual_with` so velocity (or divergence, etc.) is computed only once per iteration.

### What the optional knobs are for

Every step has the four mandatory pieces (`residual`, `update`, `control`, plus a `name`) and an arbitrary number of optional modifiers. None of these are needed for the simplest cases; each one is added when a real method demands it:

| modifier | what it solves |
|---|---|
| `mask: <state attr>`        | apply only on the icemask (or a derived mask) — zero residual elsewhere |
| `cadence: <years>`          | apply only every N years (typical for friction inversions on a slow clock) |
| `start_time` / `end_time`   | activate the step only during a time window (e.g. start friction at $t = 10$ yr after geometry has stabilised) |
| `update.r_max`              | clip the residual to a stable range — prevents the inversion from taking large steps in noisy regions |
| `update.apply: per_step \| per_application` | $\Delta t = $ outer-loop dt vs. $\Delta t = 1$ — selects whether $\alpha$ is a per-time rate or a per-application gain |
| `smoother.sigma`            | mask-aware Gaussian low-pass on the residual — useful when the observation is noisy (e.g. velocity log-ratio for friction inversions) |
| `control.bounds: [lo, hi]`  | scalar clip on the resulting control |
| `control.floor_at` / `ceil_at: <state attr>` | per-pixel clip against another state field — e.g. `floor_at: topg` enforces $z_s \geq z_b$ |
| `control.outside_mask: <c>` | constant fill outside the mask (legacy `out_of_mass_smb = -10`) |
| `geometry_policy`           | when the control is one of `(thk, topg, usurf)`: how to restore $z_s = z_b + h$ after writing one of them |
| `shares_residual_with: <other step>` | reuse a residual already computed by another step (saves recomputing `divflux` or `velsurf_mag`) |

In addition, `pre_processes` and `post_processes` lists let the user place auxiliary modules at the right phase of the inner loop (see *How the inner loop is structured* below).

---

## How the inner loop is structured

Each iteration:

```
1. pre_processes.update          (e.g. effective_pressure, smb_simple)
2. forward_model.update          (default: iceflow)
3. advance time                  (CFL-limited dt + save-time alignment)
4. for each due step:
       r = residual(state)              # T - M, log(M/T), etc.
       r = (r * mask) smoothed?
       ΔC = update_law(C, r, dt_eff)    # additive / multiplicative / replace
       C ← clip(new_C, bounds, floor_at, ceil_at, outside_mask)
       apply geometry_policy if writing thk/topg/usurf
5. post_processes.update         (e.g. thk for mass conservation)
6. snapshot                      (output hooks + misfit CSV)
```

`pre_processes` produce fields the forward model reads (effective pressure, SMB, …); `post_processes` consume the controls just written and use the save-aligned `state.dt` (typically `thk` for mass conservation).

A **step** is a triple plus optional modifiers:

| component | values |
|---|---|
| `residual.kind`   | `linear` $r = T-M$ • `relative` $r = (T-M)/\max(\lvert T \rvert, \varepsilon)$ • `log_ratio` $r = \log(\max(M,\varepsilon)/\max(T,\varepsilon))$ |
| `update.kind`     | `additive` $C \leftarrow C + \alpha r \Delta t$ • `multiplicative` $C \leftarrow C \exp(\alpha r \Delta t)$ • `multiplicative_linear` $C \leftarrow C(1 + \alpha r \Delta t)$ • `replace` $C \leftarrow \alpha r$ |
| `update.apply`    | `per_step` (uses `state.dt`) • `per_application` (uses $\Delta t = 1$, default when `cadence > 0`) |
| `control.field`   | any state attribute (typically `smb`, `slidingco`, `topg`, `thk`, `usurf`) |
| `geometry_policy` | `none` • `recompute_usurf` ($\text{usurf} = \text{topg} + \text{thk}$) • `recompute_topg` ($\text{topg} = \text{usurf} - \text{thk}$, then thk re-clipped) |

Optional modifiers per step: `mask`, `cadence`, `start_time`, `end_time`, `smoother.sigma` (mask-aware Gaussian, TF-only), `update.r_max` (residual clip), `control.bounds` (scalar clip), `control.floor_at` / `control.ceil_at` (per-pixel field clip), `control.outside_mask` (constant fill outside the mask), `shares_residual_with` (reuse another step's residual computation).

Steps are independent. A single config can run **several steps in parallel** in the same time loop — typically a geometry-inversion step plus a friction-inversion step — sharing the time clock and the forward-model call. Two steps can share a residual via `shares_residual_with` so the divergence (or velocity, etc.) is computed only once per iteration.

---

## Recovering classical methods from this schema

### PISM-style force-to-thickness

Drift `smb` so that ice thickness relaxes to a target:

| component | value |
|---|---|
| residual       | `linear`, target `thk_target`, current `thk` |
| update         | `replace`, `alpha: α` |
| control        | `smb`, with `bounds: [smb_min, smb_max]` |
| post_processes | `[thk]` (mass conservation does the actual relaxation) |

This produces the closed-form $H(t) = H_\text{target} + (H_0 - H_\text{target})\exp(-\alpha t)$.

### Apparent-mass-balance bed inversion (Frank & van Pelt, 2025)

Drive flux divergence toward `amb = smb − dhdt_obs` by perturbing thickness and surface elevation jointly. Two geometry steps share one residual:

| step | control | $\alpha$ |
|---|---|---|
| `amb_thk`   | `thk`   | $\beta$ |
| `amb_usurf` | `usurf` | $\theta \beta$ |

```yaml
defaults:
  - override /processes:
    - iceflow
    - smb_simple              # produces state.smb
  - override /assimilations:
    - time_relaxation

processes:
  smb_simple:
    update_freq: 1.0
    array:
      - ["time", "gradabl", "gradacc", "ela", "accmax"]
      - [0, 0.007, 0.004, 3050, 2.0]

assimilations:
  time_relaxation:
    forward_model: iceflow
    pre_processes: [smb_simple]
    steps:
      - name: amb_thk
        residual: { kind: linear, target: amb, current: divflux }
        update:   { kind: additive, alpha: 0.138 }   # β
        control:  { field: thk, bounds: [0.0, 2000.0] }
        mask: icemask
        geometry_policy: recompute_topg

      - name: amb_usurf
        shares_residual_with: amb_thk
        residual: { kind: linear, target: amb, current: divflux }
        update:   { kind: additive, alpha: 0.0422 }  # θ·β
        control:  { field: usurf, floor_at: topg }
        mask: icemask
        geometry_policy: recompute_topg
```

The target `amb` is recognised by `_ensure_derived`, which computes `state.amb = state.smb − state.dhdt_obs` (masked by `icemask`) at every iteration. `state.dhdt_obs` is snapshotted from the input `dhdt` field at startup. The legacy combined parameter $\theta$ disappears as a free knob — it is just the ratio `α_usurf / α_thk`.

### Linear-multiplicative friction (legacy IGM additive)

Multiply `slidingco` by $(1 + \mathrm{clip}(r))$ where $r$ is the relative velocity mismatch — the legacy IGM friction-inversion kernel:

| component | value |
|---|---|
| residual    | `relative`, target `velsurf_magobs`, current `velsurf_mag` |
| update      | `multiplicative_linear`, `alpha: -1`, `r_max: max_vel_ratio`, `apply: per_application` |
| control     | `slidingco`, with `bounds` |

The `α = -1` flips the sign because the schema's `relative` residual is $(T-M)/T = (\text{obs}-\text{mod})/\text{obs}$ while the legacy formula is $(\text{mod}-\text{obs})/\text{obs}$.

---

## Diagnostics

A CSV log of step-level residual norms is written at every save time:

```yaml
time_relaxation:
  outputs:
    misfits:
      path: misfits.csv
      track:
        - { step: amb_thk,  kind: rmse }
        - { step: friction, kind: rmse }
```

Standard IGM output modules (`write_ncdf`, `write_ts`, `write_vtp`) are also called at every save time inside the inner loop.

---

## Parameters

Default configuration file ([time_relaxation.yaml](https://github.com/instructed-glacier-model/igm/blob/main/igm/conf/assimilations/time_relaxation.yaml)):

~~~yaml
{% include  "../../../../igm/conf/assimilations/time_relaxation.yaml" %}
~~~

{% set config = load_yaml('../igm/conf/assimilations/time_relaxation.yaml') %}
{% set help = load_yaml('../igm/conf_help/assimilations/time_relaxation.yaml') %}
{% set module_key = config.keys() | list | first %}
{% set module = config[module_key] %}
{% set module_help = help %}

{% include "includes/_config_table_tree.j2" %}
