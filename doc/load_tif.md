### <h1 align="center" id="title">IGM module load_tif </h1>

# Description:

This IGM module loads spatial 2D raster data from any tif file present in the working directory folder, and transform each of them into tensorflow variables, the name of the file becoming the name of the variable, e.g. the file topg.tif will yield variable topg, ect... It is expected here to import at least basal topography (variable `topg`). It also complete the data, e.g. the basal topography from ice thickness and surface topography. Note that all these variables will therefore be available in the code with `state.myvar` from myvar.tif (e.g. variable `icemask` can be provided, and served to define an accumulation area -- this is usefull for modelling an individual glaciers, and prevent overflowing in neighbouring catchements). The module also contains two functions for resampling (parameter `ltif_coarsen` should be increased to 2,3,4 ..., default 1 value means no coarsening) and cropping the data (parameter `ltif_crop` should be set to True, and the bounds must be definined as wished).

This module depends on `rasterio`.
 
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
||`--ltif_coarsen`|`1`|coarsen the data to a coarser resolution (default: 1), e.g. 2 would be twice coarser ignore data each 2 grid points|
||`--ltif_crop`|`False`|Crop the data with xmin, xmax, ymin, ymax (default: False)|
||`--ltif_xmin`|`-100000000000000000000`|crop_xmin|
||`--ltif_xmax`|`100000000000000000000`|crop_xmax|
||`--ltif_ymin`|`-100000000000000000000`|crop_ymin|
||`--ltif_ymax`|`100000000000000000000`|crop_ymax|
