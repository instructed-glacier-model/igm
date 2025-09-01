# store fixtures here

empty_core_expected = """
cwd: ${get_cwd:0}
core:
  saved_params_filename: params_saved
  url_data: ''
  folder_data: data
  logging_file: ''
  logging: false
  logging_level: 30
  gpu_id: 0
  gpu_info: true
  print_params: true
modules:
  load_ncdf:
    lncd_input_file: input.nc
    lncd_method_coarsen: skipping
    lncd_coarsen: 1
    lncd_crop: false
    lncd_xmin: -(10**20)
    lncd_xmax: 10**20
    lncd_ymin: -(10**20)
    lncd_ymax: 10**20
  smb_simple:
    smb_simple_update_freq: 1.0
    smb_simple_file: smb_simple_param.txt
    smb_simple_array: []
  iceflow:
    iceflow:
      iflo_run_pretraining: false
      iflo_run_data_assimilation: false
      iflo_type: emulated
      iflo_pretrained_emulator: true
      iflo_emulator: ''
      iflo_init_slidingco: 0.0464
      iflo_init_arrhenius: 78
      iflo_enhancement_factor: 1.0
      iflo_regu_glen: 1.0e-05
      iflo_regu_weertman: 1.0e-10
      iflo_exp_glen: 3
      iflo_exp_weertman: 3
      iflo_gravity_cst: 9.81
      iflo_ice_density: 910
      iflo_new_friction_param: true
      iflo_save_model: false
      iflo_Nz: 10
      iflo_vert_spacing: 4.0
      iflo_thr_ice_thk: 0.1
      iflo_solve_step_size: 1
      iflo_solve_nbitmax: 100
      iflo_solve_stop_if_no_decrease: true
      iflo_fieldin:
      - thk
      - usurf
      - arrhenius
      - slidingco
      - dX
      iflo_dim_arrhenius: 2
      iflo_retrain_emulator_freq: 10
      iflo_retrain_emulator_lr: 2.0e-05
      iflo_retrain_emulator_nbit_init: 1
      iflo_retrain_emulator_nbit: 1
      iflo_retrain_emulator_framesizemax: 750
      iflo_multiple_window_size: 0
      iflo_force_max_velbar: 0
      iflo_network: cnn
      iflo_activation: LeakyReLU
      iflo_nb_layers: 16
      iflo_nb_blocks: 4
      iflo_nb_out_filter: 32
      iflo_conv_ker_size: 3
      iflo_dropout_rate: 0
      iflo_weight_initialization: glorot_uniform
      iflo_exclude_borders: 0
      iflo_cf_eswn: []
      iflo_cf_cond: false
      iflo_regu: 0.0
      iflo_min_sr: 1.0e-20
      iflo_max_sr: 1.0e+20
      iflo_force_negative_gravitational_energy: false
      iflo_optimizer_solver: Adam
      iflo_optimizer_lbfgs: false
      iflo_optimizer_emulator: Adam
      iflo_save_cost_emulator: ''
      iflo_save_cost_solver: ''
      iflo_output_directory: ''
    optimize:
      opti_vars_to_save:
      - usurf
      - thk
      - slidingco
      - velsurf_mag
      - velsurfobs_mag
      - divflux
      - icemask
      opti_init_zero_thk: 'False'
      opti_regu_param_thk: 10.0
      opti_regu_param_slidingco: 1
      opti_regu_param_arrhenius: 10.0
      opti_regu_param_div: 1
      opti_smooth_anisotropy_factor: 0.2
      opti_smooth_anisotropy_factor_sl: 1.0
      opti_convexity_weight: 0.002
      opti_convexity_power: 1.3
      opti_usurfobs_std: 2.0
      opti_velsurfobs_std: 1.0
      opti_thkobs_std: 3.0
      opti_divfluxobs_std: 1.0
      opti_divflux_method: upwind
      opti_force_zero_sum_divflux: 'False'
      opti_scaling_thk: 2.0
      opti_scaling_usurf: 0.5
      opti_scaling_slidingco: 0.0001
      opti_scaling_arrhenius: 0.1
      opti_control:
      - thk
      opti_cost:
      - velsurf
      - thk
      - icemask
      opti_nbitmin: 50
      opti_nbitmax: 500
      opti_step_size: 1
      opti_step_size_decay: 0.9
      opti_output_freq: 50
      opti_save_result_in_ncdf: geology-optimized.nc
      opti_plot2d_live: true
      opti_plot2d: true
      opti_save_iterat_in_ncdf: true
      opti_editor_plot2d: vs
      opti_uniformize_thkobs: true
      sole_mask: false
      opti_retrain_iceflow_model: true
      opti_to_regularize: topg
      opti_include_low_speed_term: false
      opti_infer_params: false
      opti_tidewater_glacier: false
      opti_vol_std: 1000.0
      fix_opti_normalization_issue: false
    pretraining:
      data_dir: surflib3d_shape_100
      batch_size: 1
      freq_test: 20
      train_iceflow_emulator_restart_lr: 2500
      epochs: 5000
      min_arrhenius: 5
      max_arrhenius: 151
      min_slidingco: 0
      max_slidingco: 20000
      min_coarsen: 0
      max_coarsen: 2
      soft_begining: 500
  time:
    time_start: 2000.0
    time_end: 2100.0
    time_save: 10.0
    time_cfl: 0.3
    time_step_max: 1.0
  thk:
    thk_slope_type: superbee
    thk_ratio_density: 0.91
    thk_default_sealevel: 0.0
  print_info:
    dummy_input: 0
experiment: {}
"""

override_core_expected = """
core:
  saved_params_filename: params_saved
  url_data: ''
  folder_data: data_folder
  logging_file: logging file directory
  logging: true
  logging_level: 10
  gpu_id: 2
  gpu_info: true
  print_params: true
"""

override_modules_expected = """
modules:
  load_ncdf:
    lncd_input_file: input.nc
    lncd_method_coarsen: skipping
    lncd_coarsen: 1
    lncd_crop: false
    lncd_xmin: -(10**20)
    lncd_xmax: 10**20
    lncd_ymin: -(10**20)
    lncd_ymax: 10**20
  smb_simple:
    smb_simple_update_freq: 1.0
    smb_simple_file: smb_simple_param.txt
    smb_simple_array: []
  iceflow:
    iceflow:
      iflo_run_pretraining: false
      iflo_run_data_assimilation: false
      iflo_type: emulated
      iflo_pretrained_emulator: true
      iflo_emulator: ''
      iflo_init_slidingco: 0.0464
      iflo_init_arrhenius: 78
      iflo_enhancement_factor: 1.0
      iflo_regu_glen: 1.0e-05
      iflo_regu_weertman: 1.0e-10
      iflo_exp_glen: 3
      iflo_exp_weertman: 3
      iflo_gravity_cst: 9.81
      iflo_ice_density: 910
      iflo_new_friction_param: true
      iflo_save_model: false
      iflo_Nz: 10
      iflo_vert_spacing: 4.0
      iflo_thr_ice_thk: 0.1
      iflo_solve_step_size: 1
      iflo_solve_nbitmax: 100
      iflo_solve_stop_if_no_decrease: true
      iflo_fieldin:
      - thk
      - usurf
      - arrhenius
      - slidingco
      - dX
      iflo_dim_arrhenius: 2
      iflo_retrain_emulator_freq: 10
      iflo_retrain_emulator_lr: 2.0e-05
      iflo_retrain_emulator_nbit_init: 1
      iflo_retrain_emulator_nbit: 1
      iflo_retrain_emulator_framesizemax: 750
      iflo_multiple_window_size: 0
      iflo_force_max_velbar: 0
      iflo_network: cnn
      iflo_activation: LeakyReLU
      iflo_nb_layers: 16
      iflo_nb_blocks: 4
      iflo_nb_out_filter: 32
      iflo_conv_ker_size: 3
      iflo_dropout_rate: 0
      iflo_weight_initialization: glorot_uniform
      iflo_exclude_borders: 0
      iflo_cf_eswn: []
      iflo_cf_cond: false
      iflo_regu: 0.0
      iflo_min_sr: 1.0e-20
      iflo_max_sr: 1.0e+20
      iflo_force_negative_gravitational_energy: false
      iflo_optimizer_solver: Adam
      iflo_optimizer_lbfgs: false
      iflo_optimizer_emulator: Adam
      iflo_save_cost_emulator: ''
      iflo_save_cost_solver: ''
      iflo_output_directory: ''
    optimize:
      opti_vars_to_save:
      - usurf
      - thk
      - slidingco
      - velsurf_mag
      - velsurfobs_mag
      - divflux
      - icemask
      opti_init_zero_thk: 'False'
      opti_regu_param_thk: 10.0
      opti_regu_param_slidingco: 1
      opti_regu_param_arrhenius: 10.0
      opti_regu_param_div: 1
      opti_smooth_anisotropy_factor: 0.2
      opti_smooth_anisotropy_factor_sl: 1.0
      opti_convexity_weight: 0.002
      opti_convexity_power: 1.3
      opti_usurfobs_std: 2.0
      opti_velsurfobs_std: 1.0
      opti_thkobs_std: 3.0
      opti_divfluxobs_std: 1.0
      opti_divflux_method: upwind
      opti_force_zero_sum_divflux: 'False'
      opti_scaling_thk: 2.0
      opti_scaling_usurf: 0.5
      opti_scaling_slidingco: 0.0001
      opti_scaling_arrhenius: 0.1
      opti_control:
      - thk
      opti_cost:
      - velsurf
      - thk
      - icemask
      opti_nbitmin: 50
      opti_nbitmax: 500
      opti_step_size: 1
      opti_step_size_decay: 0.9
      opti_output_freq: 50
      opti_save_result_in_ncdf: geology-optimized.nc
      opti_plot2d_live: true
      opti_plot2d: true
      opti_save_iterat_in_ncdf: true
      opti_editor_plot2d: vs
      opti_uniformize_thkobs: true
      sole_mask: false
      opti_retrain_iceflow_model: true
      opti_to_regularize: topg
      opti_include_low_speed_term: false
      opti_infer_params: false
      opti_tidewater_glacier: false
      opti_vol_std: 1000.0
      fix_opti_normalization_issue: false
    pretraining:
      data_dir: surflib3d_shape_100
      batch_size: 1
      freq_test: 20
      train_iceflow_emulator_restart_lr: 2500
      epochs: 5000
      min_arrhenius: 5
      max_arrhenius: 151
      min_slidingco: 0
      max_slidingco: 20000
      min_coarsen: 0
      max_coarsen: 2
      soft_begining: 500
  time:
    time_start: -2000.0
    time_end: 10000.0
    time_save: 100.0
    time_cfl: 0.3
    time_step_max: 1.0
  thk:
    thk_slope_type: superbee
    thk_ratio_density: 0.91
    thk_default_sealevel: 0.0
  print_info:
    dummy_input: 0
"""

override_core_expected_cli = """
core:
  saved_params_filename: overriden_params_cli
  url_data: ''
  folder_data: data_folder
  logging_file: logging file directory
  logging: true
  logging_level: 10
  gpu_id: 3
  gpu_info: true
  print_params: true
modules:
  load_ncdf:
    lncd_input_file: input.nc
    lncd_method_coarsen: skipping
    lncd_coarsen: 1
    lncd_crop: false
    lncd_xmin: -(10**20)
    lncd_xmax: 10**20
    lncd_ymin: -(10**20)
    lncd_ymax: 10**20
  smb_simple:
    smb_simple_update_freq: 1.0
    smb_simple_file: smb_simple_param.txt
    smb_simple_array: []
  iceflow:
    iceflow:
      iflo_run_pretraining: false
      iflo_run_data_assimilation: false
      iflo_type: emulated
      iflo_pretrained_emulator: true
      iflo_emulator: ''
      iflo_init_slidingco: 0.0464
      iflo_init_arrhenius: 78
      iflo_enhancement_factor: 1.0
      iflo_regu_glen: 1.0e-05
      iflo_regu_weertman: 1.0e-10
      iflo_exp_glen: 3
      iflo_exp_weertman: 3
      iflo_gravity_cst: 9.81
      iflo_ice_density: 910
      iflo_new_friction_param: true
      iflo_save_model: false
      iflo_Nz: 10
      iflo_vert_spacing: 4.0
      iflo_thr_ice_thk: 0.1
      iflo_solve_step_size: 1
      iflo_solve_nbitmax: 100
      iflo_solve_stop_if_no_decrease: true
      iflo_fieldin:
      - thk
      - usurf
      iflo_dim_arrhenius: 2
      iflo_retrain_emulator_freq: 10
      iflo_retrain_emulator_lr: 2.0e-05
      iflo_retrain_emulator_nbit_init: 1
      iflo_retrain_emulator_nbit: 1
      iflo_retrain_emulator_framesizemax: 750
      iflo_multiple_window_size: 0
      iflo_force_max_velbar: 0
      iflo_network: cnn
      iflo_activation: LeakyReLU
      iflo_nb_layers: 16
      iflo_nb_blocks: 4
      iflo_nb_out_filter: 32
      iflo_conv_ker_size: 3
      iflo_dropout_rate: 0
      iflo_weight_initialization: glorot_uniform
      iflo_exclude_borders: 0
      iflo_cf_eswn: []
      iflo_cf_cond: false
      iflo_regu: 0.0
      iflo_min_sr: 1.0e-20
      iflo_max_sr: 1.0e+20
      iflo_force_negative_gravitational_energy: false
      iflo_optimizer_solver: Adam
      iflo_optimizer_lbfgs: false
      iflo_optimizer_emulator: Adam
      iflo_save_cost_emulator: ''
      iflo_save_cost_solver: ''
      iflo_output_directory: ''
    optimize:
      opti_vars_to_save:
      - usurf
      - thk
      - slidingco
      - velsurf_mag
      - velsurfobs_mag
      - divflux
      - icemask
      opti_init_zero_thk: 'False'
      opti_regu_param_thk: 10.0
      opti_regu_param_slidingco: 1
      opti_regu_param_arrhenius: 10.0
      opti_regu_param_div: 1
      opti_smooth_anisotropy_factor: 0.2
      opti_smooth_anisotropy_factor_sl: 1.0
      opti_convexity_weight: 0.002
      opti_convexity_power: 1.3
      opti_usurfobs_std: 2.0
      opti_velsurfobs_std: 1.0
      opti_thkobs_std: 3.0
      opti_divfluxobs_std: 1.0
      opti_divflux_method: upwind
      opti_force_zero_sum_divflux: 'False'
      opti_scaling_thk: 2.0
      opti_scaling_usurf: 0.5
      opti_scaling_slidingco: 0.0001
      opti_scaling_arrhenius: 0.1
      opti_control:
      - thk
      opti_cost:
      - velsurf
      - thk
      - icemask
      opti_nbitmin: 50
      opti_nbitmax: 500
      opti_step_size: 1
      opti_step_size_decay: 0.9
      opti_output_freq: 50
      opti_save_result_in_ncdf: geology-optimized.nc
      opti_plot2d_live: true
      opti_plot2d: true
      opti_save_iterat_in_ncdf: true
      opti_editor_plot2d: vs
      opti_uniformize_thkobs: true
      sole_mask: false
      opti_retrain_iceflow_model: true
      opti_to_regularize: topg
      opti_include_low_speed_term: false
      opti_infer_params: false
      opti_tidewater_glacier: false
      opti_vol_std: 1000.0
      fix_opti_normalization_issue: false
    pretraining:
      data_dir: surflib3d_shape_100
      batch_size: 1
      freq_test: 20
      train_iceflow_emulator_restart_lr: 2500
      epochs: 5000
      min_arrhenius: 5
      max_arrhenius: 151
      min_slidingco: 0
      max_slidingco: 20000
      min_coarsen: 0
      max_coarsen: 2
      soft_begining: 500
  time:
    time_start: 2000.0
    time_end: 2100.0
    time_save: 10.0
    time_cfl: 0.3
    time_step_max: 1.0
  thk:
    thk_slope_type: superbee
    thk_ratio_density: 0.91
    thk_default_sealevel: 0.0
  print_info:
    dummy_input: 0
"""

overriden_defaults_list_expected = """
cwd: ${get_cwd:0}
core:
  saved_params_filename: params_saved
  url_data: ''
  folder_data: data_folder
  logging_file: logging file directory
  logging: true
  logging_level: 10
  gpu_id: 8
  gpu_info: true
  print_params: true
modules:
  time:
    time_start: 2000.0
    time_end: 2100.0
    time_save: 10.0
    time_cfl: 0.3
    time_step_max: 1.0
  thk:
    thk_slope_type: superbee
    thk_ratio_density: 0.91
    thk_default_sealevel: 0.0
"""
