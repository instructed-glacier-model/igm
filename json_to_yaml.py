import json
import yaml
import pandas as pd
import copy

# === Load correspondence mapping ===
#correspondence_df = pd.read_csv("iceflow_parameter_correspondence_0.csv")
#correspondence_map = dict(zip(correspondence_df["old"], correspondence_df["new"]))

correspondence_map = {'iflo_run_data_assimilation': 'iceflow.run_data_assimilation', 'iflo_type': 'iceflow.method', 'iflo_force_max_velbar': 'iceflow.force_max_velbar', 'iflo_gravity_cst': 'iceflow.physics.gravity_cst', 'iflo_ice_density': 'iceflow.physics.ice_density', 'iflo_init_slidingco': 'iceflow.physics.init_slidingco', 'iflo_init_arrhenius': 'iceflow.physics.init_arrhenius', 'iflo_enhancement_factor': 'iceflow.physics.enhancement_factor', 'iflo_exp_glen': 'iceflow.physics.exp_glen', 'iflo_exp_weertman': 'iceflow.physics.exp_weertman', 'iflo_regu_glen': 'iceflow.physics.regu_glen', 'iflo_regu_weertman': 'iceflow.physics.regu_weertman', 'iflo_new_friction_param': 'iceflow.physics.new_friction_param', 'iflo_dim_arrhenius': 'iceflow.physics.dim_arrhenius', 'iflo_regu': 'iceflow.physics.regu', 'iflo_thr_ice_thk': 'iceflow.physics.thr_ice_thk', 'iflo_min_sr': 'iceflow.physics.min_sr', 'iflo_max_sr': 'iceflow.physics.max_sr', 'iflo_force_negative_gravitational_energy': 'iceflow.physics.force_negative_gravitational_energy', 'iflo_cf_eswn': 'iceflow.physics.cf_eswn', 'iflo_cf_cond': 'iceflow.physics.cf_cond', 'iflo_Nz': 'iceflow.numerics.Nz', 'iflo_vert_spacing': 'iceflow.numerics.vert_spacing', 'iflo_solve_step_size': 'iceflow.solver.step_size', 'iflo_solve_nbitmax': 'iceflow.solver.nbitmax', 'iflo_solve_stop_if_no_decrease': 'iceflow.solver.stop_if_no_decrease', 'iflo_optimizer_solver': 'iceflow.solver.optimizer', 'iflo_optimizer_lbfgs': 'iceflow.solver.lbfgs', 'iflo_save_cost_solver': 'iceflow.solver.save_cost', 'iflo_fieldin': 'iceflow.emulator.fieldin', 'iflo_retrain_emulator_freq': 'iceflow.emulator.retrain_freq', 'iflo_retrain_emulator_lr': 'iceflow.emulator.lr', 'iflo_retrain_emulator_lr_init': 'iceflow.emulator.lr_init', 'iflo_retrain_warm_up_it': 'iceflow.emulator.warm_up_it', 'iflo_retrain_emulator_nbit_init': 'iceflow.emulator.nbit_init', 'iflo_retrain_emulator_nbit': 'iceflow.emulator.nbit', 'iflo_retrain_emulator_framesizemax': 'iceflow.emulator.framesizemax', 'iflo_run_pretraining': 'iceflow.emulator.run_pretraining', 'iflo_pretrained_emulator': 'iceflow.emulator.pretrained', 'iflo_emulator': 'iceflow.emulator.name', 'iflo_save_model': 'iceflow.emulator.save_model', 'iflo_exclude_borders': 'iceflow.emulator.exclude_borders', 'iflo_optimizer_emulator': 'iceflow.emulator.optimizer', 'iflo_optimizer_emulator_clipnorm': 'iceflow.emulator.optimizer_clipnorm', 'iflo_optimizer_emulator_epsilon': 'iceflow.emulator.optimizer_epsilon', 'iflo_save_cost_emulator': 'iceflow.emulator.save_cost', 'iflo_output_directory': 'iceflow.emulator.output_directory', 'iflo_network': 'iceflow.emulator.network.architecture', 'iflo_multiple_window_size': 'iceflow.emulator.network.multiple_window_size', 'iflo_activation': 'iceflow.emulator.network.activation', 'iflo_nb_layers': 'iceflow.emulator.network.nb_layers', 'iflo_nb_blocks': 'iceflow.emulator.network.nb_blocks', 'iflo_nb_out_filter': 'iceflow.emulator.network.nb_out_filter', 'iflo_conv_ker_size': 'iceflow.emulator.network.conv_ker_size', 'iflo_dropout_rate': 'iceflow.emulator.network.dropout_rate', 'iflo_weight_initialization': 'iceflow.emulator.network.weight_initialization', 'opti_control': 'data_assimilation.control_list', 'opti_cost': 'data_assimilation.cost_list', 'opti_nbitmin': 'data_assimilation.optimization.nbitmin', 'opti_nbitmax': 'data_assimilation.optimization.nbitmax', 'opti_step_size': 'data_assimilation.optimization.step_size', 'opti_step_size_decay': 'data_assimilation.optimization.step_size_decay', 'opti_init_zero_thk': 'data_assimilation.optimization.init_zero_thk', 'opti_uniformize_thkobs': 'data_assimilation.fitting.uniformize_thkobs', 'opti_sole_mask': 'data_assimilation.optimization.sole_mask', 'opti_retrain_iceflow_model': 'data_assimilation.optimization.retrain_iceflow_model', 'opti_include_low_speed_term': 'data_assimilation.fitting.include_low_speed_term', 'opti_fix_opti_normalization_issue': 'data_assimilation.optimization.fix_opti_normalization_issue', 'opti_velsurfobs_thr': 'data_assimilation.fitting.velsurfobs_thr', 'opti_log_slidingco': 'data_assimilation.fitting.log_slidingco', 'opti_regu_param_thk': 'data_assimilation.regularization.thk', 'opti_regu_param_slidingco': 'data_assimilation.regularization.slidingco', 'opti_regu_param_arrhenius': 'data_assimilation.regularization.arrhenius', 'opti_regu_param_div': 'data_assimilation.regularization.divflux', 'opti_smooth_anisotropy_factor': 'data_assimilation.regularization.smooth_anisotropy_factor', 'opti_smooth_anisotropy_factor_sl': 'data_assimilation.regularization.smooth_anisotropy_factor_sl', 'opti_convexity_weight': 'data_assimilation.regularization.convexity_weight', 'opti_convexity_power': 'data_assimilation.regularization.convexity_power', 'opti_to_regularize': 'data_assimilation.regularization.to_regularize', 'opti_usurfobs_std': 'data_assimilation.fitting.usurfobs_std', 'opti_velsurfobs_std': 'data_assimilation.fitting.velsurfobs_std', 'opti_thkobs_std': 'data_assimilation.fitting.thkobs_std', 'opti_divfluxobs_std': 'data_assimilation.fitting.divfluxobs_std', 'opti_divflux_method': 'data_assimilation.divflux.method', 'opti_force_zero_sum_divflux': 'data_assimilation.divflux.force_zero_sum', 'opti_scaling_thk': 'data_assimilation.scaling.thk', 'opti_scaling_usurf': 'data_assimilation.scaling.usurf', 'opti_scaling_slidingco': 'data_assimilation.scaling.slidingco', 'opti_scaling_arrhenius': 'data_assimilation.scaling.arrhenius', 'opti_output_freq': 'data_assimilation.output.freq', 'opti_plot2d_live': 'data_assimilation.output.plot2d_live', 'opti_plot2d': 'data_assimilation.output.plot2d', 'opti_save_result_in_ncdf': 'data_assimilation.output.save_result_in_ncdf', 'opti_save_iterat_in_ncdf': 'data_assimilation.output.save_iterat_in_ncdf', 'opti_editor_plot2d': 'data_assimilation.output.editor_plot2d', 'opti_vars_to_save': 'data_assimilation.output.vars_to_save', 'opti_infer_params': 'data_assimilation.cook.infer_params', 'opti_tidewater_glacier': 'data_assimilation.cook.tidewater_glacier', 'opti_vol_std': 'data_assimilation.cook.vol_std', 'data_dir': 'pretraining.data_dir', 'batch_size': 'pretraining.batch_size', 'freq_test': 'pretraining.freq_test', 'train_iceflow_emulator_restart_lr': 'pretraining.train_iceflow_emulator_restart_lr', 'epochs': 'pretraining.epochs', 'min_arrhenius': 'pretraining.min_arrhenius', 'max_arrhenius': 'pretraining.max_arrhenius', 'min_slidingco': 'pretraining.min_slidingco', 'max_slidingco': 'pretraining.max_slidingco', 'min_coarsen': 'pretraining.min_coarsen', 'max_coarsen': 'pretraining.max_coarsen', 'soft_begining': 'pretraining.soft_begining'}

#translate_df = pd.read_csv("module_prefix_correspondence.csv", header=None)
#translate_map = dict(zip(translate_df.iloc[:,0], translate_df.iloc[:,1]))

translate_map= {'time': 'time', 'thk': 'thk', 'avalanche': 'avalanche', 'load_ncdf': 'lncd', 'load_tif': 'ltif', 'plot2d': 'plt2d', 'clim_glacialindex': 'clim', 'clim_oggm': 'clim_oggm', 'smb_accpdd': 'smb_accpdd', 'smb_accmelt': 'smb_accmelt', 'smb_oggm': 'smb_oggm', 'smb_simple': 'smb_simple', 'vert_flow': 'vflo', 'enthalpy': 'enth', 'write_ncdf': 'wncd', 'read_output': 'rncd', 'oggm_shop': 'oggm', 'particles': 'part', 'gflex': 'gflex'}

def remove_comments(json_str):
    lines = json_str.split("\n")
    cleaned_lines = [line for line in lines if not line.strip().startswith(("//", "#"))]
    return "\n".join(cleaned_lines)

def flatten_to_nested(flat_dict):
    nested = {}
    for key, value in flat_dict.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = value
    return nested

def apply_parameter_mapping(params):
    new_params = {}
    for key, val in params.items():
        if key in correspondence_map:
            new_key = correspondence_map[key]
            new_params[new_key] = val
        else:
            new_params[key] = val
    return new_params

# === Load JSON ===
with open("params.json", "r") as f:
    json_text = remove_comments(f.read())
params_dict = json.loads(json_text)

if "iflo_run_data_assimilation" in params_dict:
    del params_dict["iflo_run_data_assimilation"]
    params_dict["modules_preproc"].append("optimize")
 
# Extract modules
imodules = params_dict.pop("modules_preproc", [])
modules = params_dict.pop("modules_process", [])
omodules = [m for m in params_dict.pop("modules_postproc", []) if m not in ["print_info", "print_comp"]]

# Apply param mapping
params_mapped = apply_parameter_mapping(params_dict)

# Flatten to nested dictionary
nested_params = flatten_to_nested(params_mapped)

Imodules = copy.deepcopy(imodules)
Modules = copy.deepcopy(modules)
Omodules = copy.deepcopy(omodules)

if 'optimize' in Imodules:
    Imodules.remove('optimize')
    Modules.insert(0, 'data_assimilation')

if 'oggm_shop' in Imodules:
    Imodules.insert(1, 'local')

if 'icemasktxt' in Imodules:
    Imodules.remove('optimize')

# === Generate YAML ===
yaml_structure = {
    "core": {
        "url_data": nested_params.get("url_data", "")
    },
    "defaults": [
        {"override /inputs": Imodules},
        {"override /processes": Modules},
        {"override /outputs": Omodules}
    ],
    "inputs": {},
    "processes": {},
    "outputs": {}
}

# Remove core key from nested
nested_params.pop("url_data", None)
 
def assign_params(section, module_list, raw_params):
    for mod in module_list:
        if mod == "iceflow": 
            iceflow_params = {correspondence_map[k]: v for k, v in raw_params.items() if k in correspondence_map}
            iceflow_nested = flatten_to_nested(iceflow_params)
            if iceflow_nested:
                section[mod] = iceflow_nested.get("iceflow", {})
        if mod == "optimize": 
            iceflow_params = {correspondence_map[k]: v for k, v in raw_params.items() if k in correspondence_map}
            iceflow_nested = flatten_to_nested(iceflow_params)
            if iceflow_nested:
                section[mod] = iceflow_nested.get("data_assimilation", {})
        if mod == "pretraining": 
            iceflow_params = {correspondence_map[k]: v for k, v in raw_params.items() if k in correspondence_map}
            iceflow_nested = flatten_to_nested(iceflow_params)
            if iceflow_nested:
                section[mod] = iceflow_nested.get("pretraining", {})
        elif mod in translate_map:
            prefix = translate_map[mod]
            filtered = {k[len(prefix) + 1:]: v for k, v in raw_params.items() if k.startswith(prefix + "_")}
            if filtered:
                section[mod] = filtered
        else:
            if mod not in section:
                continue


assign_params(yaml_structure["inputs"], imodules, params_dict)
assign_params(yaml_structure["processes"], modules, params_dict)
assign_params(yaml_structure["outputs"], omodules, params_dict)

if "optimize" in yaml_structure.get("inputs", {}):
    yaml_structure["processes"]["data_assimilation"] = copy.deepcopy(yaml_structure["inputs"]["optimize"])
    del yaml_structure["inputs"]["optimize"]

# === Write YAML ===
with open("params.yaml", "w") as f:
    f.write("# @package _global_\n\n")

    yaml_text = yaml.dump(yaml_structure, sort_keys=False)
    top_keys = ["core:", "defaults:", "inputs:", "processes:", "outputs:"]
    lines = yaml_text.splitlines()
    final_lines = []

    for i, line in enumerate(lines):
        if any(line.startswith(k) for k in top_keys):
            if i > 0:
                final_lines.append("")  # insert blank line before top-level section
        final_lines.append(line)

    f.write("\n".join(final_lines) + "\n")
