# Thickness-evolution architecture

The public `thk.py` module is an orchestrator. Numerical transport, optional
front evolution, active-domain constraints, boundary conditions, and surface
reconstruction are separate concerns:

```text
thk.py                  lifecycle, component selection, composition
  -> transport/         transport implementations, dispatch table, selection
  -> fronts/            calving-front methods, dispatch table, selection
       common.py        representation, transport step, rules shared by the methods
       sub_grid.py      Albrecht et al. (2011) / PISM sub-grid front
       level_set.py     mass-consistent level-set front
  -> domains/           composable active-cell constraints, selection
  -> boundary.py        the shared boundary contract
  -> sources.py         the source term smb + bmb passed to every backend
  -> masks.py           flotation and the Q1 ice masks shared with the ice flow
  -> rigid_body.py      removal of mechanically unanchored ice
  -> surfaces.py        reconstruct lsurf/usurf after the update
```

Every extension point has the same shape: a dispatch dictionary, and a
`get_*(cfg)` function returning plain data (`transport.get_transport`,
`fronts.get_front`, `domains.get_domain_constraints`). Selection logic
therefore lives next to the backends it selects, and adding a backend never
touches `thk.py`.

A component is just a module exposing `initialize(cfg, state)` and
`update(cfg, state)`, optionally `finalize(cfg, state)` — the same convention
an IGM process module follows, so no wrapper type is needed.

`thk.py` holds only the module lifecycle and composition. The dispatch
dictionaries remain beside their implementations, in the same direct style as
`iceflow.vertical.VerticalDiscrs`. `_select_components(cfg)` turns the
configuration into a validated set of components and `initialize` stores it on
`state.thk_components`, following the process-owned container pattern used by
the iceflow module. This deliberately small dataclass contains run-level
orchestration data, not spatial model fields. `update` only reads that
container, so the configuration is parsed once per run rather than once per
timestep:

```python
state.thk_components.transport_name         # configured transport name
state.thk_components.transport              # selected transport module
state.thk_components.front                  # selected front module, or None
state.thk_components.pipeline               # ordered modules that actually run
state.thk_components.domain_constraints     # resolved active-domain constraints
state.thk_components.boundaries             # normalized four-side boundary data
state.thk_components.transport_options      # backend-owned static option dictionary
state.thk_components.component_state        # namespaced non-spatial bookkeeping
```

Spatial fields and diagnostics — `thk`, `divflux`, `thk_active_mask`,
`thk_initial_ice_mask`, and solver counters — belong directly on `state`.
This lets IGM's generic state slicing, memory reporting, and explicitly
configured outputs recognize them. Small Python-only backend state is kept in
the component's namespaced entry of `component_state`.

Only components that actually execute are initialized. In particular, a
front that owns transport does not initialize an otherwise unused transport
backend. The same ordered `pipeline` drives initialization and updates, and
finalization traverses it in reverse. This avoids maintaining parallel
`mass_transport`, `front_after_transport`, and initialization records.

## Backend status

| backend | status | large timestep | principal limitation |
| --- | --- | --- | --- |
| `explicit` | production default | no | advective CFL |
| `implicit_x` | production, specialized | yes | independent x-flowlines only |
| `ffsl` | opt-in/beta | yes | deformation-controlled internal substeps |
| `implicit` | opt-in/beta | yes | first-order upwind/backward-Euler diffusion |
| `adi` | experimental | conditionally | non-monotone at large CFL |

The calving fronts (`sub_grid`, `level_set`) own transport through the
explicit scheme's divergence and are described below.

Selection, composition, and validation are one function rather than one per
component, because they are not independent: the front is only compatible with
certain transport schemes, and whichever component ends up owning mass
transport is the one that has to support active domains.

## Adding a transport backend

A backend exposes `initialize(cfg, state)` and `update(cfg, state)`. Add its
module to the `TransportSchemes` dictionary in `transport/__init__.py`,
following the direct dispatch pattern used by
`iceflow.vertical.VerticalDiscrs`. The root package deliberately exposes only
the IGM lifecycle functions. No registration call or change to `thk.py` is
needed. The update publishes at least `state.thk` and `state.divflux`.

A backend declares what it supports, and an omitted declaration always means
"not supported", so a new scheme can never silently ignore an option:

| declaration | omitted means |
| --- | --- |
| `SUPPORTED_BOUNDARY_MODES` | only the historical `zero` boundary |
| `SUPPORTS_ACTIVE_DOMAIN` | a restricted `domain.constraints` is rejected |
| `SUPPORTS_DIVFLUX_SMOOTHING` | `divflux_smooth_sigma` > 0 is rejected |

These are checked once, centrally, against whichever component ends up owning
mass transport — so a `replace_transport` front is held to the same contract
as a transport scheme.

Tensor-only numerical work belongs in a small number of compiled kernels. The
transport adapters cache Python configuration in a plain dictionary at
initialization and unpack it before passing tensors plus static boundary
strings to compiled `tf.function` kernels; dictionaries never cross a tracing
boundary. FFSL and implicit iteration uses `tf.while_loop`, without host
convergence checks.

## Active domains

`domain.constraints` is a list whose entries are intersected.  Each entry
selects one backend from `domains.DomainConstraints`; adding a new constraint
requires one module and one dictionary entry.  The resulting
`state.thk_active_mask` is also available to timestep controllers. An empty
constraint list is a strict no-op: the mask is not allocated, and explicit
transport plus CFL selection use their historical unmasked paths. Transport
backends must explicitly declare `SUPPORTS_ACTIVE_DOMAIN` before a restricted
domain can use them, so a new scheme cannot silently ignore internal no-flux
edges or fixed cells.

## Calving fronts

`front.method` is `none`, `sub_grid` or `level_set`. The rate at which a
front retreats relative to the ice, `a = c + m_cf` (calving plus frontal
melt, m/yr), comes from the `calving_rate` process
(`state.calving_rate`, `state.frontal_melt_rate`); rules that prescribe the
front position are front options (`min_thickness`, the thickness rule of
Albrecht et al., 2011; `fixed`, no advance beyond the initial front).

**Representation.** `state.thk` is the one, physical thickness (volume per
area) and the field the ice flow reads. A node carries ice only once its
finite-volume cell is full, so the stress-balance front is sharp on the
boundary of full Q1 cells: the ice flow's active cells are the full-ice
cells, plus partially covered cells whose four corners are grounded (land
margins; `masks.iceflow_node_mask`, the rule of the unified evaluator).
`state.Href` (m, area-specific volume, PISM's `ice_area_specific_volume`)
holds the ice of the *partial cells*, ice-free ocean cells next to ice. They
are invisible to the ice flow and the surfaces, take the mass balance over
their covered fraction, and the total volume is `sum(thk + Href) dx**2`.
There is no second thickness field.

**Transport step** (`fronts/common.py`, one compiled kernel). The ice flow
gives no velocity off its active nodes, so the velocity is extended two
rings into the ocean cells and orphan ice nodes (mean over the neighbours
carrying velocity; PISM's margin extrapolation; two rings so that the cells
beyond an orphan node get a velocity too), fluxes out of cells next to the
front are first-order upwind (`first_order`), and the divergence is the
explicit scheme's slope limiter with the configured boundaries. Full and
land cells advance; the inflow into a partial cell goes into its `Href`.
The flux into a partial cell is then the ice-side `u H` (Albrecht et al.,
2011, Eq. 2), which behind an advancing front equals the flux through the
shelf.

**Threshold thickness `H_r`**, the thickness a partial cell takes when it
fills (`front.threshold`): `flux` (default) gives it the thickness
continuity gives to the ice entering it, the mean over its ice neighbours of
`H_n |u_n| / |u_p|` with `|u_p|` the speed extrapolated linearly to the cell;
`mean` is PISM's mean of the neighbours. Where the bed is above the
neighbours' mean ice base, both fill to their mean surface (PISM). On a
spreading shelf the mean leaves a wall one cell thick that travels with the
front (Albrecht et al., 2011, Sect. 3), a first-order error: with the
Weertman velocity prescribed, the front after 150 yr lags by 1.9, 1.0 and
0.5 km at 5, 2.5 and 1.25 km with `mean`, against +0.6, +0.2 and +0.1 km with
`flux`, whose front thickness is within 1 % of the analytic profile
(`tests/test_thk/test_front.py`, igm-albrecht).

**sub_grid** (`fronts/sub_grid.py`, PISM `GeometryEvolution` part_grid and
`FrontRetreat`): a partial cell becomes full when `Href >= H_r`, with
`thk = H_r`; the residual is split among its ice-free ocean neighbours for
up to `max_iterations` passes of a device-side loop (`residual: discard` is
Albrecht's variant 1; `common.fill_partial_cells`). Retreat drains `dt a H_r L / dx` from each front cell
and converts the adjacent full cells when the reservoir is exhausted (at
most one cell per step; the rest is reported). `L` is the front length in
the cell, `sum |n . e_face|` over its faces with the ice (n from the Sobel
gradient of the ice fraction): 1 for a front along a grid axis, as in
PISM, and sqrt(2) across a 45-degree staircase, which removes PISM's
orientation bias (the face flux into the cell carries the same factor).

**level_set** (`fronts/level_set.py`): `psi` moves with `w = u - a n`
(Bondzio et al., 2016), upwind in device-side substeps, with the ice
velocity extended linearly into the first ring (on a spreading shelf the
ice at the front is faster than at the last full node), and the ice follows
it exactly. The fill fraction of a front cell is `clip(1/2 - psi/w)`, with
`w = dx (|n_x| + |n_y|)` its width across the front; calving removes the
ice over the area the front leaves (the column thickness of a partial cell
is kept), a partial cell full by area or volume becomes ice at `H_r` with
its residual passed to the cells the front has entered (else its ice-free
ocean neighbours), and a full cell the *ablation* enters becomes a partial
cell holding `thk phi/phi_adv`, the covered area left by the calving alone.
A trailing edge (ice advected away from an ice-free cell) is thinned once,
by the transport, and moves cell by cell as in sub_grid. `psi` is
re-distanced every step with the partial cells frozen (full and empty cells
re-clamped after), so no mass moves, and is kept consistent with the ice
(inside every full cell, up to the fill fraction of a cell that received a
residual, outside every cell the rules emptied, following the ice on land);
an empty cell next to the ice sits exactly at half its width, so that the
smallest advection can enter it (an outward margin there deadlocked slow
fronts, whose routed inflow was wiped as spurious calving). The same mass
kernels as sub_grid close the budget; on the kinematic circle test (60
steps, front CFL 0.2, advancing 3.6 km) the level-set front drifts by about
+0.5 dx, against -0.2 dx for sub_grid.

**Rules and clean-up**, both methods: `min_thickness` calves floating front
cells thinner than it, at any ice/ocean interface -- holes and rifts
included, as PISM's `next_to_ice_free_ocean` (the `ocean_connected_only`
option of `calving_rate` does not reach this rule); `fixed` calves ice
beyond the initial front; `Href`
merges into `thk` where both are positive or the cell is no longer below the
water level, and is calved where no ice is left next to it; with
`remove_rigid_body_modes`, filling nodes next to anchored ice are kept.

**Diagnostics:** `state.calved_thk` (m removed from each cell during the
last step, including rigid-body removal), `state.calving_unapplied_thk`,
`state.ice_area_fraction` (1 full, the fill fraction, 0), and `state.psi`
(level set). Every front step closes the mass budget:
`d sum(thk + Href) = dt sum(source) - boundary outflux - sum(calved_thk)`,
with the source over the full cells plus the partial cells' covered
fraction, up to two known approximations: a sink (melt) exceeding the ice
present is clipped uncounted, and inflow through a Dirichlet ghost into an
ice-free marine edge cell is dropped (see `_route`).

The time step includes `c + m_cf` in its CFL condition when the
`calving_rate` process runs, so the front retreats at most `cfl` cells per
step relative to the ice.

**Cost** (RTX 4090, float32, one thk update with a front): sub_grid 3.8 ms
at 200 x 350 and 4.3 ms at 1000 x 1000, level_set 5.8 and 6.1 ms (the former
sub_grid: 60 and 62 ms), against 2 ms for the explicit transport alone. The
cost is dominated by kernel dispatch, not by the grid size.

**Composition.** Both methods declare `UPDATE_MODE = "replace_transport"`,
`COMPATIBLE_TRANSPORTS = ("explicit",)` (a scheme that moves ice through a
partial cell within one step would disperse the front) and the explicit
boundary modes; active-domain constraints and flux smoothing are rejected.
A new front method is one module in `fronts.FrontMethods` declaring
`UPDATE_MODE`, `COMPATIBLE_TRANSPORTS`, `AVAILABLE`, `UNAVAILABLE_REASON`;
`after_transport` fronts run right after the transport scheme.

**Configuration changes (2026-09).** The former keys fail loudly:

| former `thk` key | now |
| --- | --- |
| `calving_front: true`, `method: sub_grid` | `front.method: sub_grid` |
| `method: level_set` | `front.method: level_set` (now mass-consistent and available) |
| `front_slope_type: godunov` | `front.first_order: true` |
| `interior_slope_type` | `slope_type` (the bulk limiter) |
| `only_marine` | removed: partial cells and calving are marine by construction |
| `extend_halo`, `extend_thresh` | removed: the ice flow reads `thk` itself, no padding |
| `sub_grid.href_cap_factor`, `sub_grid.calve_cliff` | removed (mass-destroying cap; double retreat) |
| `sub_grid.promote_iters` | `front.sub_grid.max_iterations` |
| `level_set.*` | `front.level_set.{reinit_freq, reinit_iter, band}` |
| `calving_rate.law: thickness_threshold`, `Hcr` | `thk.front.min_thickness` |
| `calving_rate.K2`, `c_max` | `calving_rate.eigen.K`, `calving_rate.max_rate` |
| `state.thk_true` | `state.thk` (single field); partial-cell ice in `state.Href` |

## Boundary contract

`boundary.left`, `boundary.right`, `boundary.top`, and `boundary.bottom`
independently select:

- `zero` / `open`: zero exterior ice; outflow is allowed and inflow is ice-free.
- `symmetric` / `closed` / `no_flux`: zero normal face velocity and mass flux.
- `periodic`: connect the two opposite faces, including arbitrary-CFL FFSL
  swept integrals.

Periodic sides must be paired (`left` with `right`, `top` with `bottom`). The
four-side form is the only accepted boundary configuration, preventing an
axis shorthand from accidentally imposing the same condition at physically
different boundaries such as an ice divide and a terminus.

FFSL and implicit implement all three. Explicit implements `zero`,
`symmetric` and `dirichlet` (the exterior thickness held at the initial
thickness of the edge cells: an inflow boundary) independently on all four
sides while retaining a separate, unchanged all-zero graph for the default.
ADI retains its historical zero-exterior behavior; selecting another mode
fails at initialization.

`implicit_x` is a batched flowline backend: every row is advanced independently
along x by one XLA-compiled tridiagonal solve. It supports independently open
or symmetric left/right sides, uses the shared `implicit.theta`, and contains
no y transport. Periodic x flowlines would require a cyclic rather than an
ordinary tridiagonal system and are deliberately rejected.

## Source term

Every backend advances `dH/dt + div(H u) = source` with the source returned
by `sources.mass_balance(state)`: the surface mass balance `state.smb`, plus
the basal mass balance `state.bmb` when a module publishes it (the `bmb`
process). Both are in m ice eq. yr-1 and positive for a gain. A backend never
reads `state.smb` directly, so surface and basal terms cannot be combined
differently by two schemes, and a run without `bmb` passes `state.smb` itself,
bit-identical to earlier versions. `bmb` counts only when the `bmb` process
is active (`ThkComponents.basal_mass_balance`), so a `bmb` field read from an
input file, e.g. an earlier IGM output, is never applied by mistake.

The compiled kernels take this combined field as their `source` argument. The
source enters exactly as the surface balance did before, masked outside an
active domain; with a calving front, a partial cell takes it over its covered
fraction and the open ocean takes none.

## Conservation diagnostics

For implicit theta and ADI transport, `state.thk_transport_divflux` is the raw
theta-method transport divergence. If a non-monotone theta value creates
negative thickness, the public `state.divflux` is adjusted to close the actual
post-projection thickness budget and
`state.thk_nonnegative_correction_volume` reports the added volume. With the
default backward Euler (`theta=1`) the transport alone creates no negative
thickness, so the correction is solver-scale roundoff plus any ablation
larger than the available ice (`-source * dt > H`), e.g. strong sub-shelf melt
under thin ice, which is clipped at zero.

FFSL reports unavailable ablation in `state.ffsl_source_limiter_volume`; its
public divergence likewise closes the actual nonnegative thickness update.

The iterative implicit backend defaults to `failure_policy: stop`: a failed
solve retains the old thickness, publishes `state.thk_step_accepted=false`,
and clears `state.continue_run` with tensor operations. FFSL applies the same
contract when `limit_policy: stop` and its requested deformation substeps
exceed the configured cap. Neither policy performs a Python/NumPy convergence
check or forces a device-to-host synchronization.

## Flotation consistency

`thk.ratio_density` controls floating-surface reconstruction. Iceflow uses
`physics.ice_density / physics.water_density` for its grounding and ocean
stress terms. When both are configured, thickness initialization verifies that
the ratios agree; maintaining two materially different grounding lines in one
run is rejected rather than silently accepted.
