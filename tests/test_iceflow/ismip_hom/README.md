# ISMIP-HOM Tests

Configuration-driven test framework for ISMIP-HOM experiments.

## Quick Start

```bash
# Run all enabled experiments
pytest tests/test_iceflow/ismip_hom/

# Run specific experiment
pytest tests/test_iceflow/ismip_hom/exp_a/

# Run specific test
pytest tests/test_iceflow/ismip_hom/exp_a/ -k "20 and adam"
```

Configure tests by editing [`test_config.yaml`](test_config.yaml).

## Configuration

### Experiments

```yaml
experiments: [exp_a, exp_b, exp_c, exp_d, exp_e_1]
lengths: [20, 40, 80, 160]  # For exp_a-d only
```

Only listed experiments run. exp_e_1 and exp_e_2 don't use length scales.

### Methods

```yaml
methods:
  unified:
    enabled: true
    mappings: [identity, network]
    optimizers: [adam, lbfgs, cg_newton]
    optimizer_mappings:
      cg_newton: [identity]
```

`optimizer_mappings` optionally restricts an optimizer to compatible mappings.
The CG-Newton ISMIP-HOM configuration uses the MOLHO-capable banded operator
with finite-difference probes and a vertical-component block-Jacobi
preconditioner, so it is tested with the identity mapping only.

### Validation

```yaml
validation:
  mode: compare_reference  # run_only | compare_reference
  tolerance:
    type: relative_l2      # relative_l2 | absolute_l2 | relative_max | absolute_max
    default: 0.10          # 10% default
    experiments:
      exp_e_1: 0.20        # Per-experiment override
      exp_e_2: 0.20
  save_plots: true
```

**Modes:** `run_only` (just run) or `compare_reference` (validate vs reference)

**Types:** `relative_l2` (default), `absolute_l2`, `relative_max`, `absolute_max`

### Skip tests

```yaml
skip:
  - {experiment: exp_a, length: 160, mapping: network, optimizer: adam}
```

## Examples

**Quick smoke test:**
```yaml
experiments: [exp_a]
lengths: [20]
methods:
  unified:
    mappings: [identity]
    optimizers: [adam]
```

**Strict validation:**
```yaml
validation:
  tolerance:
    default: 0.05
```

## Structure

```
ismip_hom/
├── test_config.yaml              # Configuration
├── exp_a/, exp_b/, exp_c/, exp_d/  # Experiments (with lengths)
├── exp_e_1/, exp_e_2/            # Experiments (no lengths)
│   ├── test_exp_*.py
│   ├── experiment/               # Hydra configs
│   └── outputs/                  # Results
├── data/oga/                     # Reference data
└── utils/                        # Framework
```
