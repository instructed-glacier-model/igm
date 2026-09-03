# Thickness-evolution architecture

The public `thk.py` module is an orchestrator. Numerical transport, optional
front evolution, active-domain constraints, boundary conditions, and surface
reconstruction are separate concerns:

```text
thk.py                  lifecycle, component selection, composition
  -> transport/         transport implementations, dispatch table, selection
  -> fronts/            front implementations, dispatch table, selection
       utils.py         neighbourhood helpers shared by the front backends
  -> domains/           composable active-cell constraints, selection
  -> boundary.py        the shared boundary contract
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
`state.thk_components`, following the `state.iceflow` container used by the
iceflow module. `update` then only reads that container, so the configuration
is parsed once per run rather than once per timestep:

```python
state.thk_components.transport_name         # configured transport name
state.thk_components.transport              # selected transport module
state.thk_components.front                  # selected front module, or None
state.thk_components.mass_transport         # whichever of the two moves mass
state.thk_components.front_after_transport  # a front that runs after transport
state.thk_components.domain_constraints     # resolved active-domain constraints
state.thk_components.boundaries             # normalized four-side boundary data
state.thk_components.transport_options      # backend-owned cached static data
```

A component parks its own per-run bookkeeping on that container too, rather
than on `state` (`initial_ice_mask` for the `initial_ice` constraint,
`psi_built` and `steps_since_reinit` for the level-set front). Only genuine
fields and diagnostics — `thk`, `divflux`, `thk_active_mask`, the solver
counters — belong directly on `state`, where the output modules can write
them.

Only components that actually execute are initialized. In particular, a
front that owns transport does not initialize an otherwise unused transport
backend. Finalization runs over the same components in reverse order.

## Backend status

| backend | status | large timestep | principal limitation |
| --- | --- | --- | --- |
| `explicit` | production default | no | advective CFL |
| `implicit_x` | production, specialized | yes | independent x-flowlines only |
| `ffsl` | opt-in/beta | yes | deformation-controlled internal substeps |
| `implicit` | opt-in/beta | yes | first-order upwind/backward-Euler diffusion |
| `adi` | experimental | conditionally | non-monotone at large CFL |

The `sub_grid` and unavailable `level_set` front implementations are kept
separate from this transport maturity classification. Their numerical
algorithms are intentionally unchanged by the transport cleanup.

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
transport adapters cache Python configuration at initialization and pass
tensors plus static boundary strings to compiled `tf.function` kernels. FFSL
and implicit iteration uses `tf.while_loop`, without host convergence checks.

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

## Adding front evolution

A front backend also exposes `initialize` and `update`. Add a `FrontMethod`
value to `fronts.FrontMethods`. Its `update_mode` is explicit:

- `after_transport`: run the selected transport backend, then evolve the front.
- `replace_transport`: the front backend owns mass transport as well as front
  evolution. Compatible transport scheme names must be declared so a selected
  backend is never silently ignored.

The existing sub-grid implementation is listed as `replace_transport` and
currently supports only the legacy explicit transport path. A future FFSL
partial-cell front should preferably use conservative swept face masses and
can then either become a composable `after_transport` component or declare a
dedicated FFSL transport contract.

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

FFSL and implicit implement all three. Explicit implements `zero` and
`symmetric` independently on all four sides while retaining a separate,
unchanged all-zero graph for the default. ADI retains its historical
zero-exterior behavior; selecting another mode fails at initialization.

`implicit_x` is a batched flowline backend: every row is advanced independently
along x by one XLA-compiled tridiagonal solve. It supports independently open
or symmetric left/right sides, uses the shared `implicit.theta`, and contains
no y transport. Periodic x flowlines would require a cyclic rather than an
ordinary tridiagonal system and are deliberately rejected.

## Conservation diagnostics

For implicit theta and ADI transport, `state.thk_transport_divflux` is the raw
theta-method transport divergence. If a non-monotone theta value creates
negative thickness, the public `state.divflux` is adjusted to close the actual
post-projection thickness budget and
`state.thk_nonnegative_correction_volume` reports the added volume. With the
default backward Euler (`theta=1`) this correction should be zero apart from
solver-scale roundoff.

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
