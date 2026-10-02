import os
import shutil
import warnings

import numpy as np
import tensorflow as tf
import xarray as xr
from netCDF4 import Dataset
from xarray.conventions import encode_cf_variable

from igm.utils.math.getmag import getmag
from igm.utils.ncdf_dims import (
    vertical_dim_name,
    vertical_dims_in_use,
    vertical_size_reference,
)

# Lossless codecs for `compression`. Only zlib is built into every netCDF-C; the
# others need its HDF5 filter plugins, both to write and to read the file. The
# blosc codecs are left out: their HDF5 filter fails the write of any chunk it
# cannot shrink, instead of storing it uncompressed.
NETCDF_CODECS = ("zlib", "zstd", "bzip2")


def initialize(cfg, state):
    state.var_info_ncdf_ex = {
        "topg": ["Basal Topography", "m"],
        "usurf": ["Surface Topography", "m"],
        "thk": ["Ice Thickness", "m"],
        "icemask": ["Ice mask", "NO UNIT"],
        "smb": ["Surface Mass Balance", "m/y ice eq"],
        "ubar": ["x depth-average velocity of ice", "m/y"],
        "vbar": ["y depth-average velocity of ice", "m/y"],
        "velbar_mag": ["Depth-average velocity magnitude of ice", "m/y"],
        "uvelsurf": ["x surface velocity of ice", "m/y"],
        "vvelsurf": ["y surface velocity of ice", "m/y"],
        "wvelsurf": ["z surface velocity of ice", "m/y"],
        "velsurf_mag": ["Surface velocity magnitude of ice", "m/y"],
        "uvelbase": ["x basal velocity of ice", "m/y"],
        "vvelbase": ["y basal velocity of ice", "m/y"],
        "wvelbase": ["z basal velocity of ice", "m/y"],
        "velbase_mag": ["Basal velocity magnitude of ice", "m/y"],
        "divflux": ["Divergence of the ice flux", "m/y"],
        "strflowctrl": ["arrhenius+1.0*slidingco", "MPa$^{-3}$ a$^{-1}$"],
        "dtopgdt": ["Erosion rate", "m/y"],
        "arrhenius": ["Arrhenius factor", "MPa$^{-3}$ a$^{-1}$"],
        "slidingco": ["Reference basal shear stress (legacy stack)", "MPa"],
        "tau_ref": ["Reference basal shear stress", "MPa"],
        "meantemp": ["Mean annual surface temperatures", "°C"],
        "meanprec": ["Mean annual precipitation", "Kg m^(-2) y^(-1)"],
        "velsurfobs_mag": ["Obs. surf. speed of ice", "m/y"],
        "weight_particles": ["weight_particles", "no"],
        "T": ["Ice temperature", "K"],
        "omega": ["Water content fraction", "1"],
        "E": ["Ice enthalpy", "J kg-1"],
        "E_pmp": ["Pressure melting point enthalpy", "J kg-1"],
        "T_pmp": ["Pressure melting point temperature", "K"],
        "T_pa": ["Pressure-adjusted temperature", "K"],
        "T_pa_b": ["Pressure-adjusted temperature at bed", "K"],
        "E_s": ["Surface enthalpy BC", "J kg-1"],
        "T_s": ["Surface temperature", "K"],
        "basal_melt_rate": ["Basal melt rate (enthalpy)", "m/y ice eq"],
        "bmb": ["Basal Mass Balance", "m/y ice eq"],
        "shelf_melt_rate": ["Sub-shelf melt rate", "m/y ice eq"],
        "grounded_fraction": ["Grounded fraction of the cell", "1"],
        "ocean_temp": ["Ocean temperature at the ice base", "°C"],
        "ocean_salinity": ["Ocean salinity at the ice base", "g/kg"],
        "ocean_thermal_forcing": ["Ocean thermal forcing", "K"],
        "pico_box": ["PICO box number", "1"],
        "grounding_line_depth": ["Grounding-line depth of the plume", "m"],
    }

    state.var_info_ncdf_ts = {}
    state.var_info_ncdf_ts["vol"] = ["Ice volume", "km^3"]
    state.var_info_ncdf_ts["area"] = ["Glaciated area", "km^2"]

    # NetCDF files written so far (path -> encoding and open handle), and Zarr stores
    state.local_netcdf_files = {}
    state.local_zarr_stores = set()

    check_compression(cfg)

    if "zarr" in cfg.outputs.local.file_format_list:
        try:
            import zarr  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "outputs.local: the 'zarr' format needs the optional `zarr` package "
                "(pip install zarr)."
            ) from e


def run(cfg, state):

    if not state.saveresult:
        if not getattr(state, "continue_run", False):
            close_files(state)
        return

    # Prepare any derived quantities
    if "velbar_mag" in cfg.outputs.local.vars_to_save:
        state.velbar_mag = getmag(state.ubar, state.vbar)

    if "velsurf_mag" in cfg.outputs.local.vars_to_save:
        state.velsurf_mag = getmag(state.uvelsurf, state.vvelsurf)

    if "velbase_mag" in cfg.outputs.local.vars_to_save:
        state.velbase_mag = getmag(state.uvelbase, state.vvelbase)

    file_format_list = cfg.outputs.local.file_format_list

    # One xr.Dataset per save, shared by every format written from it
    if "netcdf" in file_format_list or "zarr" in file_format_list:
        ds = dataset_ex(cfg, state)

        if "netcdf" in file_format_list:
            write_netcdf(cfg, state, ds, cfg.outputs.local.output_file)

        if "zarr" in file_format_list:
            write_zarr(state, ds, zarr_path(cfg.outputs.local.output_file))

    if "tif" in file_format_list:
        write_tif(cfg, state)

    if cfg.outputs.local.write_ts:
        write_netcdf(cfg, state, dataset_ts(state), cfg.outputs.local.output_ts_file)

    if not getattr(state, "continue_run", False):
        close_files(state)


#############################################


def write_tif(cfg, state):

    var_list = cfg.outputs.local.vars_to_save

    for var in var_list:
        if not hasattr(state, var):
            continue

        var_data = getattr(state, var).numpy()
        file_name = (
            f"{var}-{str(getattr(state, 't', tf.constant(0)).numpy()).zfill(6)}.tif"
        )

        data_array = xr.DataArray(
            var_data,
            dims=("y", "x"),
            coords={"y": state.y.numpy(), "x": state.x.numpy()},
        )

        if "crs" in cfg.outputs.local:
            data_array.rio.write_crs(cfg.outputs.local.crs, inplace=True)

        data_array.rio.to_raster(file_name)


#####################################


def dataset_ex(cfg, state):
    """The fields of `vars_to_save` at the current time, as a one-step xr.Dataset."""

    nz_ref = vertical_size_reference(cfg)

    data_vars = {}
    for var in cfg.outputs.local.vars_to_save:
        if not hasattr(state, var):
            continue
        arr = getattr(state, var).numpy()
        if arr.ndim == 2:
            dims = ("y", "x")
        elif arr.ndim == 3:
            dims = (vertical_dim_name(arr.shape[0], nz_ref), "y", "x")
        else:
            raise ValueError(
                f"Cannot write '{var}' to NetCDF: expected a 2-D (y, x) or 3-D "
                f"(z, y, x) field, got shape {arr.shape}."
            )
        attrs = {}
        if var in state.var_info_ncdf_ex:
            attrs["long_name"], attrs["units"] = state.var_info_ncdf_ex[var]
        # an xr.Variable rather than a DataArray: the same dataset, built 2-3x faster
        data_vars[var] = xr.Variable(("time",) + dims, arr[np.newaxis], attrs)

    coords = {
        "x": ("x", state.x.numpy(), {"units": "m", "long_name": "x", "axis": "X"}),
        "y": ("y", state.y.numpy(), {"units": "m", "long_name": "y", "axis": "Y"}),
        "time": (
            "time",
            [getattr(state, "t", tf.constant(0)).numpy()],
            {"units": "yr", "long_name": "time", "axis": "T"},
        ),
    }

    if nz_ref is not None:
        coords["z"] = ("z", np.arange(nz_ref))

    # One coordinate per vertical dimension actually present in the data. A run may
    # carry several: iceflow and enthalpy use independent vertical grids, so fields
    # on the enthalpy grid get their own dimension rather than being forced onto
    # `z` (which would make xarray reject the whole dataset over conflicting sizes).
    for name, size in vertical_dims_in_use(data_vars).items():
        coords[name] = (name, np.arange(size))

    return xr.Dataset(
        data_vars=data_vars,
        coords=coords,
        attrs={"pyproj_srs": getattr(state, "pyproj_srs", "")},
    )


def dataset_ts(state):
    """Ice volume and glaciated area at the current time, as a one-step xr.Dataset."""

    # A calving front keeps the ice of its partial cells in Href.
    vol = np.sum(state.thk + getattr(state, "Href", 0.0)) * (state.dx**2) / 10**9
    area = np.sum(state.thk > 1) * (state.dx**2) / 10**6

    def attrs(var):
        long_name, units = state.var_info_ncdf_ts[var]
        return {"long_name": long_name, "units": units}

    return xr.Dataset(
        {
            "time": (
                "time",
                [getattr(state, "t", tf.constant(0)).numpy()],
                {"units": "yr", "long_name": "time"},
            ),
            "vol": ("time", [vol], attrs("vol")),
            "area": ("time", [area], attrs("area")),
        },
        attrs={
            "vol_long_name": state.var_info_ncdf_ts["vol"][0],
            "vol_units": state.var_info_ncdf_ts["vol"][1],
            "area_long_name": state.var_info_ncdf_ts["area"][0],
            "area_units": state.var_info_ncdf_ts["area"][1],
        },
    )


#########################################################


def write_netcdf(cfg, state, ds, file_path):
    """Write one save, `ds` (a single time step), to the NetCDF file `file_path`.

    xarray writes the first save, which fixes the layout of the file: coordinates,
    attributes and encoding, with `time` unlimited. xarray cannot append along a
    dimension of an existing NetCDF file (`to_netcdf(mode="a")` overwrites the
    variables instead), so each later save is still CF-encoded by xarray, and
    netCDF4, the library under xarray's netcdf4 engine, only writes it at the next
    time index. A save therefore never reads back what the file already holds: its
    cost in time and memory does not grow with the length of the run.
    """

    entry = state.local_netcdf_files.get(file_path)

    if entry is None:
        if hasattr(state, "logger"):
            state.logger.info(f"Creating new NetCDF file {file_path} with xarray")

        encoding = netcdf_encoding(cfg, ds)
        ds.to_netcdf(file_path, mode="w", unlimited_dims=["time"], encoding=encoding)
        entry = {"encoding": encoding, "handle": None}
        state.local_netcdf_files[file_path] = entry

        # Holding the file from the first save on keeps another program from
        # opening it for reading first and so blocking the next append.
        if cfg.outputs.local.keep_open:
            try:
                entry["handle"] = open_for_append(file_path)
            except OSError:
                pass  # retried at the next save
        return

    if hasattr(state, "logger"):
        state.logger.info(
            f"Appending to NetCDF file {file_path} at iteration {state.it}"
        )

    nc = entry["handle"]
    if nc is None:
        try:
            nc = open_for_append(file_path)
        except OSError:
            # HDF5 refuses to open a file for writing while another program (a
            # viewer, a notebook) holds it open for reading.
            append_to_copy(state, ds, file_path, entry["encoding"])
            return

    append_slice(nc, ds, entry["encoding"], file_path)

    if cfg.outputs.local.keep_open:
        nc.sync()  # flush each save: a crash loses at most the one in progress
        entry["handle"] = nc
    else:
        nc.close()


def netcdf_encoding(cfg, ds):
    """One chunk per save, so each append writes (and compresses) only its slice."""

    local_cfg = cfg.outputs.local

    compression = {}
    if local_cfg.complevel > 0:
        compression = dict(
            compression=local_cfg.compression,
            complevel=local_cfg.complevel,
            shuffle=True,
        )

    encoding = {}
    for name, var in ds.data_vars.items():
        if "time" not in var.dims or var.ndim < 2:
            continue
        encoding[name] = {"chunksizes": (1,) + var.shape[1:], **compression}
        # netCDF-C quantizes on every write, appends included
        if local_cfg.significant_digits is not None and var.dtype.kind == "f":
            encoding[name]["significant_digits"] = local_cfg.significant_digits

    return encoding


def check_compression(cfg):
    """Fail at startup, not at the first save, if netCDF-C lacks the chosen codec."""

    local_cfg = cfg.outputs.local
    if local_cfg.complevel <= 0 or local_cfg.compression == "zlib":
        return

    codec = local_cfg.compression
    if codec not in NETCDF_CODECS:
        raise ValueError(
            f"outputs.local.compression: '{codec}' is not one of "
            f"{', '.join(NETCDF_CODECS)}."
        )

    with Dataset("check_compression.nc", "w", diskless=True) as nc:
        available = getattr(nc, f"has_{codec}_filter")()
    if not available:
        raise ValueError(
            f"outputs.local.compression: this netCDF-C build has no '{codec}' filter; "
            "use 'zlib', or point HDF5_PLUGIN_PATH to the HDF5 filter plugins."
        )


def open_for_append(file_path):

    nc = Dataset(file_path, "a")

    # Values arrive CF-encoded by xarray already: netCDF4 must not scale or mask them.
    nc.set_auto_maskandscale(False)

    # Each chunk holds one save and is written once, so caching it would only cost
    # memory: netCDF-C reserves up to 64 MB per variable while a file stays open.
    for var in nc.variables.values():
        if "time" in var.dimensions and var.ndim > 1:
            var.set_var_chunk_cache(size=0)

    return nc


def append_slice(nc, ds, encoding, file_path):
    """Write the variables of `ds` that depend on time at the next index of `nc`."""

    d = nc.dimensions["time"].size
    for name, var in ds.variables.items():
        if "time" not in var.dims:
            continue
        if name not in nc.variables:
            warnings.warn(
                f"outputs.local: '{name}' did not exist at the first save to "
                f"{file_path}, so it is not saved."
            )
            continue
        var = var.copy(deep=False)
        var.encoding = encoding.get(name, {})
        nc.variables[name][d, ...] = encode_cf_variable(var, name=name).values[0]


def append_to_copy(state, ds, file_path, encoding):
    """Append `ds` to a copy of `file_path`, then put the copy in its place.

    For when another program holds the file open. The copy streams through the disk,
    never through memory; the other program keeps reading the file it opened, and
    later saves append in place again (to the copy, which nothing holds open).
    """

    if hasattr(state, "logger"):
        state.logger.warning(
            f"{file_path} is open in another program: appending to a copy instead"
        )

    tmp_path = f"{file_path}.tmp"
    shutil.copyfile(file_path, tmp_path)
    nc = open_for_append(tmp_path)
    append_slice(nc, ds, encoding, file_path)
    nc.close()
    os.replace(tmp_path, file_path)


def zarr_path(output_file):
    """The Zarr store is written next to the NetCDF file: output.nc -> output.zarr."""

    return os.path.splitext(output_file)[0] + ".zarr"


def write_zarr(state, ds, store):
    """Write one save, `ds`, to the Zarr store `store`, which xarray can append to.

    Metadata is consolidated once, at the end of the run (`close_files`): doing it at
    every save rescans the whole store, which makes each save slower than the last.
    """

    if store not in state.local_zarr_stores:
        encoding = {
            name: {"chunks": (1,) + var.shape[1:]}
            for name, var in ds.data_vars.items()
            if "time" in var.dims
        }
        ds.to_zarr(store, mode="w", encoding=encoding, consolidated=False)
        state.local_zarr_stores.add(store)
    else:
        static = [name for name, var in ds.variables.items() if "time" not in var.dims]
        ds.drop_vars(static).to_zarr(
            store, mode="a", append_dim="time", consolidated=False
        )


def close_files(state):
    """End of the run: close NetCDF files kept open, consolidate Zarr metadata."""

    for entry in state.local_netcdf_files.values():
        if entry["handle"] is not None:
            entry["handle"].close()
            entry["handle"] = None

    if state.local_zarr_stores:
        import zarr

        for store in state.local_zarr_stores:
            zarr.consolidate_metadata(store)
