import xarray as xr
import numpy as np
import tensorflow as tf 

from .include_icemask import include_icemask

def run(cfg, state):

    if cfg.inputs.local.type == "netcdf":
        filepath = state.original_cwd.joinpath(cfg.core.folder_data, cfg.inputs.local.filename)
        with xr.open_dataset(
            filepath
        ) as f:
            ds = f.load()

        if "time" in ds.dims:
            if hasattr(state,'logger'):
                state.logger.info(
                    f"Time dimension found. Selecting the first time step at {cfg.processes.time.start}"
                )
            ds = ds.sel(time=ds.time[ds.time==cfg.processes.time.start])

    elif cfg.inputs.local.type == "tif":
        import rioxarray
        from pathlib import Path
 
        # Base folder where the .tif files are stored
        data_folder = Path(state.original_cwd.joinpath(cfg.core.folder_data))

        # Find all .tif files in the folder
        tif_files = list(data_folder.glob("*.tif"))

        # Create dataset by reading each .tif and naming the variable after the file stem
        ds = xr.Dataset({
            tif.stem: rioxarray.open_rasterio(tif).squeeze("band")
            for tif in tif_files
        })

    else:
        raise ValueError(f"Unknown type {cfg.inputs.local.type}")

    ds = ds.sortby(['x', 'y']) # Sort by x and y to ensure that the data is in the correct order
 
    ds = xr.where(ds > 1.0e+35, np.nan, ds)

    crop = np.any(list(dict(cfg.inputs.local.crop).values()))
    if crop:
        if hasattr(state,'logger'):
            state.logger.info("Cropping dataset")
        ds = ds.sel(
            x=slice(cfg.inputs.local.crop.xmin, cfg.inputs.local.crop.xmax),
            y=slice(cfg.inputs.local.crop.ymin, cfg.inputs.local.crop.ymax),
        )

    if cfg.inputs.local.coarsening.ratio > 1:
        if hasattr(state,'logger'):
            state.logger.info("Coarsening dataset")
        ds = ds.coarsen(
            x=cfg.inputs.local.coarsening.ratio,
            y=cfg.inputs.local.coarsening.ratio,
            boundary=cfg.inputs.local.coarsening.boundary,
        ).mean()

    # Interpolating needed?
    # ds_test = ds.interp(x=2, method="linear")

    # Example on how to convert units with xarray! 'data' is an xarray dataset...
    # x = data.Geopotential_height_isobaric.metpy.x.metpy.convert_units('meter').values
 
    ds = complete_data(ds)

    for variable, array in ds.data_vars.items():
        if (array.ndim>0)|(variable in ["dx", "dy"]):
            setattr(state, variable, tf.Variable(np.squeeze(array).astype("float32")))

    for coord, array in ds.coords.items():
        setattr(
            state, coord, tf.constant(array.astype("float32"))
        ) 

    # This is to be used to forward meta data to the output
    state.ds_meta_only = xr.Dataset(attrs=ds.attrs)  

    # dt = xr.DataTree(name="root", dataset=ds)

    if cfg.inputs.local.icemask.include:
        include_icemask(state, mask_shapefile=cfg.inputs.local.icemask.shapefile, 
                               mask_invert=cfg.inputs.local.icemask.invert)


def complete_data(ds: xr.Dataset) -> xr.Dataset:

    # ? Are units automatically included in some of these or not? I think no...
    X, Y = np.meshgrid(ds.x, ds.y)
 
    ds["dx"] = abs(ds.x.data[1] - ds.x.data[0])
    ds["dy"] = abs(ds.y.data[1] - ds.y.data[0])
    ds["X"] = xr.DataArray(X, dims=["y", "x"])
    ds["Y"] = xr.DataArray(Y, dims=["y", "x"])
    ds["dX"] = xr.DataArray(np.ones_like(X) * ds.dx.values, dims=["y", "x"])
    ds["dY"] = xr.DataArray(np.ones_like(Y) * ds.dy.values, dims=["y", "x"])

    if "thk" not in ds.data_vars:
        ds["thk"] = xr.DataArray(np.zeros_like(X), dims=["y", "x"])

    topg_exists = "topg" in ds.data_vars
    usurf_exists = "usurf" in ds.data_vars
    if not topg_exists and not usurf_exists:
        raise ValueError("Either 'topg' or 'usurf' must be present in the dataset.")

    if "topg" not in ds.data_vars:
        # ? Should we only add pixels that do not have a nan value?
        ds["topg"] = ds["usurf"] - ds["thk"]
    elif "usurf" not in ds.data_vars:
        ds["usurf"] = ds["topg"] + ds["thk"]

    return ds
