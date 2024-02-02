### <h1 align="center" id="title">IGM smb_oggm module  </h1>

# Description:

Module `smb_oggm` implements the monthly temperature index model calibrated on geodetic MB data (Hugonnet, 2021) by OGGM. The yearly surface mass balance  is computed with 
$$
SMB = \frac{\rho_w}{\rho_i}  \sum_{i=1}^{12} \left( P_i^{sol} - d_f \max \{ T_i - T_{melt}, 0 \} \right),
$$
where $P_i^{sol}$ is the is the monthly solid precipitation, $T_i$ is the monthly temperature and $T_{melt}$ is the air temperature above which ice melt is assumed to occur (parameter `temp_melt`), $d_f$ is the melt factor (parameter `melt_f`), and $\frac{\rho_w}{\rho_i} $ is the ratio of water to ice density. Solid precipitation $P_i^{sol}$ is computed out of precipitation and temperature such that it equals precipitation when the temperature is lower than a certain threshold (parameter `temp_all_solid`), zero above another threshold (parameter `temp_all_liq`), with a linear transition between the two. Module `oggm_shop` provides all calibrated parameters. 
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
||`--smb_oggm_update_freq`|`1`|Update the mass balance each X years |
||`--smb_oggm_ice_density`|`910.0`|Density of ice for conversion of SMB into ice equivalent|
||`--smb_oggm_wat_density`|`1000.0`|Density of water|
