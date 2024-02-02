### <h1 align="center" id="title">IGM module oggm_shop </h1>

# Description:

This IGM module uses OGGM utilities and GlaThiDa dataset to prepare data 
for the IGM model for a specific glacier given the RGI ID (parameter `oggm_RGI_ID`), check at [GLIMS VIeWER](https://www.glims.org/maps/glims) to find the RGI ID of your glacier. By default, data are already posprocessed (parameter `oggm_preprocess` is True), with spatial resolution of 100 and a oggm_border size of 30. For custom spatial resolution and size of 'oggm_border' to keep a safe distance to the glacier margin, one need to set `oggm_preprocess` parameter to False, and set `oggm_dx` and `oggm_border` parameter as desired. 

The module directly provide IGM with all 2D gridded variables (as tensorflow object), and are accessible in the code with e.g. `state.thk`. By default a copy of all the data are stored in a NetCDF file `input_saved.nc` so that this file can be readed directly in a second run with module `load_ncdf` instead of re-downloading the data with `oggm_shop` again. The module provides all data variables necessary to run IGM for a forward glacier evolution run (assuming we provide basal topography `topg` and ice thickness `thk`), or a preliminary data assimilation/ inverse modelling with the `optimize` module further data (typically `icemaskobs`, `thkinit`, `uvelsurf`, `vvelsurf`, `thkobs`, `usurfobs`).

Data are currently based on COPERNIUS DEM 90 for the top surface DEM, the surface ice velocity from (Millan, 2022), the ice thickness from (Millan, 2022) or (farinotti2019) (the user can choose wit parameter `oggm_thk_source` between `millan_ice_thickness` or `consensus_ice_thickness` dataset). 

When activating `oggm_include_glathida` to True, ice thickness profiles are downloaded from the [GlaThiDa depo](https://gitlab.com/wgms/glathida) and are rasterized with name `thkobs` (pixels without data are set to NaN values.).

The OGGM script was written by Fabien Maussion. The GlaThiDa script was written by Ethan Welty & Guillaume Jouvet.

The module depends (of course) of `oggm` library. Unfortunatly the module works on linux and Max, but not on windows (unless using WSL).
 
# Parameters: 


|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--working_dir`|``|Working directory (default empty string)|
||`--modules_preproc`|`['oggm_shop']`|List of pre-processing modules|
||`--modules_process`|`['iceflow', 'time', 'thk']`|List of processing modules|
||`--modules_postproc`|`['write_ncdf', 'plot2d', 'print_info']`|List of post-processing modules|
||`--logging`||Activate the looging|
||`--logging_file`|``|Logging file name, if empty it prints in the screen|
||`--print_params`||Print definitive parameters in a file for record|
||`--oggm_RGI_ID`|`RGI60-11.01450`|RGI ID|
||`--oggm_preprocess`||Use preprocessing|
||`--oggm_dx`|`100`|Spatial resolution (need preprocess false to change it)|
||`--oggm_border`|`30`|Safe border margin  (need preprocess false to change it)|
||`--oggm_thk_source`|`consensus_ice_thickness`|millan_ice_thickness or consensus_ice_thickness|
||`--oggm_vel_source`|`millan_ice_velocity`|Source of the surface velocities (millan_ice_velocity or its_live)|
||`--oggm_incl_glathida`||Make observation file (for IGM inverse)|
||`--oggm_path_glathida`|``|Path where the Glathida Folder is store, so that you don't need               to redownload it at any use of the script, if empty it will be in the home directory|
||`--oggm_save_in_ncdf`||Write prepared data into a geology file|
