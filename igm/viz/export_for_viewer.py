#!/usr/bin/env python3
"""
Export IGM netCDF outputs as a study for the IGM map viewer (https://igm-model.org/latest/viewer/).

Usage:
    igm_export_for_viewer study.yaml [--out DIR]
    igm_export_for_viewer output.nc --id 2026-Doe-TC --crs EPSG:32632 [--region NAME] [--out DIR]
    igm_export_for_viewer --serve DIR [--port 8765]
    igm_export_for_viewer --unpack DIR

Writes one Cloud-Optimized GeoTIFF per region, scenario, layer and chunk of time steps
(int16, the steps of a chunk as the samples of each pixel), packed into a container
file data.bin, and igm_viewer.json describing the study (schema igm_viewer/2). The two
files are uploaded as they are to a Zenodo record.
"""

import argparse, glob, hashlib, json, os, re, shutil, sys, unicodedata
import numpy as np
import netCDF4
import rasterio
import yaml
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.shutil import copy as rio_copy
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject, transform_bounds

NODATA = -32768
BLOCK = 256
ALIGN = 4096
MAX_BIN = 20e9

# name: (scale, colormap, log, unit, label, mask)
LAYERS = {
    "thk": (1, "viridis", False, "m", "ice thickness", "ice"),
    "usurf": (1, "gist_earth", False, "m", "surface elevation", "ice"),
    "topg": (1, "gist_earth", False, "m", "bed elevation", "domain"),
    "smb": (100, "RdBu_r", False, "m ice/yr", "surface mass balance", "ice"),
    "dhdt": (100, "RdBu_r", False, "m/yr", "thickness change rate", "ice"),
    "divflux": (100, "RdBu_r", False, "m/yr", "flux divergence", "ice"),
    "velsurf_mag": (1, "inferno", True, "m/yr", "surface speed", "ice"),
    "velbar_mag": (1, "inferno", True, "m/yr", "depth-averaged speed", "ice"),
    "velbase_mag": (1, "inferno", True, "m/yr", "basal speed", "ice"),
    "slidingco": (
        10000,
        "viridis",
        False,
        "MPa m^-1/3 yr^1/3",
        "sliding coefficient",
        "ice",
    ),
    "arrhenius": (10, "viridis", False, "MPa^-3 yr^-1", "rate factor", "ice"),
}
DERIVED = {
    "velsurf_mag": ("uvelsurf", "vvelsurf"),
    "velbar_mag": ("ubar", "vbar"),
    "velbase_mag": ("uvelbase", "vvelbase"),
}
DEFAULT_LAYERS = [
    "thk",
    "usurf",
    "topg",
    "smb",
    "velsurf_mag",
    "velbar_mag",
    "dhdt",
    "divflux",
]
DIVERGING = {"RdBu_r", "coolwarm"}


def nice(v):
    if v == 0 or not np.isfinite(v):
        return 0.0
    e = 10 ** np.floor(np.log10(abs(v)))
    for m in (1, 2, 2.5, 5, 10):
        if abs(v) <= m * e:
            return float(np.copysign(m * e, v))
    return float(v)


def colour_range(values, cmap, log):
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    if log:
        v = v[v > 0]
        if v.size == 0:
            return 1.0, 10.0
        return max(nice(np.percentile(v, 2)), 0.1), nice(np.percentile(v, 99))
    if cmap in DIVERGING:
        a = nice(np.percentile(np.abs(v), 98)) or 1.0
        return -a, a
    lo, hi = nice(np.percentile(v, 1)), nice(np.percentile(v, 99))
    if np.min(v) >= 0 and lo < 0.2 * hi:
        lo = 0.0
    return (lo, hi) if hi > lo else (lo, lo + 1.0)


def time_indices(times, sel):
    n = len(times)
    if sel in (None, "all"):
        return list(range(n))
    if isinstance(sel, str) and sel.startswith("every:"):
        step = int(sel.split(":")[1])
        idx = list(range(0, n, step))
        return idx if idx[-1] == n - 1 else idx + [n - 1]
    if isinstance(sel, (list, tuple)):
        return sorted({int(np.argmin(np.abs(times - float(t)))) for t in sel})
    raise ValueError(
        f"time_steps: expected 'all', 'every:N' or a list of times, got {sel!r}"
    )


def read_layer(ds, name, idx):
    if name in ds.variables:
        v = ds.variables[name]
        a = np.asarray(
            v[idx] if "time" in v.dimensions else v[:][None], dtype="float32"
        )
    elif name in DERIVED and all(c in ds.variables for c in DERIVED[name]):
        u, w = (
            np.asarray(ds.variables[c][idx], dtype="float32") for c in DERIVED[name]
        )
        a = np.hypot(u, w)
    else:
        return None
    fill = getattr(ds.variables.get(name), "_FillValue", None)
    if fill is not None:
        a[a == fill] = np.nan
    a[np.abs(a) > 1e30] = np.nan
    return a


class Source:
    """One netCDF output, or several on a common grid merged into a mosaic (north up)."""

    def __init__(self, files):
        files = files if isinstance(files, list) else [files]
        self.name = os.path.basename(files[0]) + (
            f" (+{len(files) - 1})" if len(files) > 1 else ""
        )
        self.ds = [netCDF4.Dataset(f) for f in files]
        d0 = self.ds[0]
        self.variables = d0.variables
        self.pyproj_srs = getattr(d0, "pyproj_srs", "")
        self.times = (
            np.asarray(d0.variables["time"][:], "float64")
            if "time" in d0.variables
            else np.array([0.0])
        )
        grids = []
        for d in self.ds:
            x = np.asarray(d.variables["x"][:], "float64")
            y = np.asarray(d.variables["y"][:], "float64")
            grids.append((x, y))
            if len(d.variables.get("time", [0])) != len(self.times):
                sys.exit(f"{self.name}: the files of a mosaic must share their times")
        self.dx = float(grids[0][0][1] - grids[0][0][0])
        x0 = min(g[0].min() for g in grids)
        y1 = max(g[1].max() for g in grids)
        nx = int(round((max(g[0].max() for g in grids) - x0) / self.dx)) + 1
        ny = int(round((y1 - min(g[1].min() for g in grids)) / self.dx)) + 1
        self.x = x0 + self.dx * np.arange(nx)
        self.y = y1 - self.dx * np.arange(ny)
        self.slots = []
        for x, y in grids:
            i = int(round((y1 - y.max()) / self.dx))
            j = int(round((x.min() - x0) / self.dx))
            if abs(x.min() - self.x[j]) > 1e-3 * self.dx:
                sys.exit(f"{self.name}: the grids of a mosaic must be aligned")
            self.slots.append((i, j, y[1] > y[0], len(y), len(x)))

    def close(self):
        for d in self.ds:
            d.close()

    def read(self, name, idx):
        if len(self.ds) == 1:
            a = read_layer(self.ds[0], name, idx)
            return a[:, ::-1, :] if a is not None and self.slots[0][2] else a
        out = None
        for d, (i, j, flip, h, w) in zip(self.ds, self.slots):
            a = read_layer(d, name, idx)
            if a is None:
                continue
            a = a[:, ::-1, :] if flip else a
            if out is None:
                out = np.full((a.shape[0], len(self.y), len(self.x)), np.nan, "float32")
            if a.shape[0] != out.shape[0]:
                a = np.broadcast_to(a[-1:], (out.shape[0], h, w))
            dst = out[:, i : i + h, j : j + w]
            np.copyto(dst, a, where=np.isnan(dst))
        return out


def layer_info(ds, name, overrides):
    if name in LAYERS:
        scale, cmap, log, unit, label, mask = LAYERS[name]
    else:
        var = ds.variables.get(name)
        scale, cmap, log, mask = None, "viridis", False, "ice"
        unit, label = getattr(var, "units", ""), getattr(var, "long_name", name)
    info = dict(scale=scale, colormap=cmap, log=log, unit=unit, label=label, mask=mask)
    info.update(overrides.get(name, {}))
    return info


def write_cog(path, data, crs, transform, descriptions):
    """Cloud-Optimized GeoTIFF, int16, the bands of `data` interleaved by pixel."""
    nb, h, w = data.shape
    profile = dict(
        driver="GTiff",
        width=w,
        height=h,
        count=nb,
        dtype="int16",
        crs=crs,
        transform=transform,
        nodata=NODATA,
    )
    with MemoryFile() as mf:
        with mf.open(**profile) as m:
            m.write(data)
            for b, d in enumerate(descriptions, start=1):
                m.set_band_description(b, d)
            rio_copy(
                m,
                path,
                driver="COG",
                BLOCKSIZE=BLOCK,
                COMPRESS="DEFLATE",
                PREDICTOR=2,
                INTERLEAVE="PIXEL",
                OVERVIEWS="AUTO",
                RESAMPLING="AVERAGE",
                BIGTIFF="IF_SAFER",
            )


WORLD_ORIGIN = (-20037508.342789244, 20037508.342789244)
WORLD_RES = 1000.0


def world_part(q_last, scale, crs, transform):
    """Last step of a layer, in physical units, on the global EPSG:3857 grid at 1 km: (row0, col0, array)."""
    h, w = q_last.shape
    b = transform_bounds(crs, "EPSG:3857", *_bounds(transform, w, h))
    c0 = int(np.floor((b[0] - WORLD_ORIGIN[0]) / WORLD_RES))
    c1 = int(np.ceil((b[2] - WORLD_ORIGIN[0]) / WORLD_RES))
    r0 = int(np.floor((WORLD_ORIGIN[1] - b[3]) / WORLD_RES))
    r1 = int(np.ceil((WORLD_ORIGIN[1] - b[1]) / WORLD_RES))
    src = np.where(q_last == NODATA, np.nan, q_last / scale).astype("float32")
    out = np.full((r1 - r0, c1 - c0), np.nan, "float32")
    reproject(
        src,
        out,
        src_transform=transform,
        src_crs=crs,
        dst_transform=from_origin(
            WORLD_ORIGIN[0] + c0 * WORLD_RES,
            WORLD_ORIGIN[1] - r0 * WORLD_RES,
            WORLD_RES,
            WORLD_RES,
        ),
        dst_crs="EPSG:3857",
        src_nodata=np.nan,
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return r0, c0, out


def assemble_world(parts, max_cells=1.2e8):
    """Merge the parts of all regions into one array, coarsened so that it stays below max_cells."""
    r0 = min(p[0] for p in parts)
    c0 = min(p[1] for p in parts)
    r1 = max(p[0] + p[2].shape[0] for p in parts)
    c1 = max(p[1] + p[2].shape[1] for p in parts)
    f = max(1, int(np.ceil(np.sqrt((r1 - r0) * (c1 - c0) / max_cells))))
    r0, c0 = r0 // f * f, c0 // f * f
    H, W = -(-(r1 - r0) // f), -(-(c1 - c0) // f)
    out = np.full((H, W), np.nan, "float32")
    for pr, pc, a in parts:
        dr, dc = pr - r0, pc - c0
        pad = np.full(
            (-(-(dr % f + a.shape[0]) // f) * f, -(-(dc % f + a.shape[1]) // f) * f),
            np.nan,
            "float32",
        )
        pad[dr % f : dr % f + a.shape[0], dc % f : dc % f + a.shape[1]] = a
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                blk = np.nanmean(
                    pad.reshape(pad.shape[0] // f, f, pad.shape[1] // f, f), axis=(1, 3)
                )
        R0, C0 = dr // f, dc // f
        dst = out[R0 : R0 + blk.shape[0], C0 : C0 + blk.shape[1]]
        m = np.isfinite(blk) & ~np.isfinite(dst)
        dst[m] = blk[m]
    res = WORLD_RES * f
    transform = from_origin(
        WORLD_ORIGIN[0] + c0 * WORLD_RES, WORLD_ORIGIN[1] - r0 * WORLD_RES, res, res
    )
    return out, transform, res


def _bounds(t, w, h):
    return (t.c, t.f + h * t.e, t.c + w * t.a, t.f)


class GridSource:
    """Static layers already on a regular north-up grid (mosaic of GeoTIFF tiles)."""

    def __init__(self, name, arrays, x, y, dx, crs):
        self.name, self.arrays = name, arrays
        self.x, self.y, self.dx, self.pyproj_srs = x, y, dx, crs
        self.times = np.array([0.0])
        self.variables = {}

    def read(self, name, idx):
        a = self.arrays.get(name)
        return None if a is None else a[None].copy()

    def close(self):
        self.arrays = {}


def _unit(lon, lat):
    lo, la = np.radians(lon), np.radians(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def _laea(xyz):
    v = xyz.mean(0)
    v /= np.linalg.norm(v)
    lat, lon = np.degrees(np.arcsin(v[2])), np.degrees(np.arctan2(v[1], v[0]))
    return CRS.from_proj4(
        f"+proj=laea +lat_0={lat:.3f} +lon_0={lon:.3f} +datum=WGS84 +units=m +no_defs"
    )


def plan_tiles(tcfg, base):
    """Group GeoTIFF tiles (one file per glacier and layer) into regions of nearby glaciers,
    each with its own equal-area grid. Returns a list of region plans."""
    from scipy.spatial import cKDTree

    pats = tcfg["layers"]
    first = next(iter(pats))
    key = lambda p: os.path.basename(p).rsplit("_", 1)[0]
    files = {
        f: {
            key(p): p
            for pat in (pat if isinstance(pat, list) else [pat])
            for p in glob.glob(os.path.join(base, pat), recursive=True)
            if p.endswith((".tif", ".tiff"))
        }
        for f, pat in pats.items()
    }
    if not files[first]:
        sys.exit(f"no file matches {pats[first]}")
    group_re = re.compile(tcfg.get("group", "^$"))
    res = float(tcfg.get("resolution", 100))
    gap = float(tcfg.get("gap_km", 150)) / 6371.0
    max_cells = float(tcfg.get("max_cells", 2e8))
    labels = {str(k): v for k, v in (tcfg.get("labels") or {}).items()}
    name_fmt = tcfg.get("name", "{group}")
    groups = {}
    for kk, p in files[first].items():
        m = group_re.search(p)
        groups.setdefault(m.group(1) if m and m.groups() else "", []).append(kk)
    plans = []
    for g in sorted(groups):
        keys = sorted(groups[g])
        wgs = []
        for kk in keys:
            with rasterio.open(files[first][kk]) as d:
                wgs.append(transform_bounds(d.crs, "EPSG:4326", *d.bounds))
        wgs = np.array(wgs)
        xyz = _unit((wgs[:, 0] + wgs[:, 2]) / 2, (wgs[:, 1] + wgs[:, 3]) / 2)
        parent = list(range(len(keys)))

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i, j in cKDTree(xyz).query_pairs(gap):
            parent[find(i)] = find(j)
        clusters = {}
        for i in range(len(keys)):
            clusters.setdefault(find(i), []).append(i)

        def grid_for(idx):
            crs = _laea(xyz[idx])
            bb = np.array([transform_bounds("EPSG:4326", crs, *wgs[i]) for i in idx])
            x0 = np.floor(bb[:, 0].min() / res) * res
            y1 = np.ceil(bb[:, 3].max() / res) * res
            nx = int(np.ceil((bb[:, 2].max() - x0) / res))
            ny = int(np.ceil((y1 - bb[:, 1].min()) / res))
            return crs, x0, y1, nx, ny, bb

        def split(idx):
            crs, x0, y1, nx, ny, bb = grid_for(idx)
            if nx * ny <= max_cells or len(idx) == 1:
                return [(idx, crs, x0, y1, nx, ny)]
            cx, cy = (bb[:, 0] + bb[:, 2]) / 2, (bb[:, 1] + bb[:, 3]) / 2
            c = cx if nx >= ny else cy
            order = np.argsort(c)
            half = len(idx) // 2
            return split([idx[i] for i in order[:half]]) + split(
                [idx[i] for i in order[half:]]
            )

        groups_idx = sorted(clusters.values(), key=len)
        merged = True
        while merged and len(groups_idx) > 1:
            merged = False
            cen = np.array([xyz[g].mean(0) for g in groups_idx])
            for i, gi in enumerate(groups_idx):
                d = np.linalg.norm(cen - cen[i], axis=1)
                d[i] = np.inf
                for j in np.argsort(d)[:3]:
                    _, _, _, nx, ny, _ = grid_for(gi + groups_idx[j])
                    if nx * ny <= max_cells:
                        groups_idx[j] = groups_idx[j] + gi
                        del groups_idx[i]
                        merged = True
                        break
                if merged:
                    break
        parts = []
        for idx in groups_idx:
            parts += split(idx)
        parts.sort(key=lambda p: -len(p[0]))
        base_name = name_fmt.format(group=g)
        for n, (idx, crs, x0, y1, nx, ny) in enumerate(parts, start=1):
            rname = base_name if len(parts) == 1 else f"{base_name}-{n}"
            label = labels.get(g, base_name)
            plans.append(
                {
                    "key": rname,
                    "label": (
                        label if len(parts) == 1 else f"{label} ({n}/{len(parts)})"
                    ),
                    "crs": crs,
                    "grid": (x0, y1, nx, ny, res),
                    "tiles": {
                        f: [files[f][keys[i]] for i in idx if keys[i] in files[f]]
                        for f in pats
                    },
                }
            )
        print(f"group {g}: {len(keys)} tiles, {len(parts)} region(s)")
    return plans


def build_tiles(plan):
    """Reproject and paste the tiles of a region plan onto its grid."""
    crs, (x0, y1, nx, ny, res) = plan["crs"], plan["grid"]
    arrays = {}
    for f, paths in plan["tiles"].items():
        arr = np.full((ny, nx), np.nan, "float32")
        for p in paths:
            with rasterio.open(p) as d:
                src = d.read(1).astype("float32")
                if d.nodata is not None:
                    src[src == d.nodata] = np.nan
                src[np.abs(src) > 1e30] = np.nan
                b = transform_bounds(d.crs, crs, *d.bounds)
                c0 = max(0, int(np.floor((b[0] - x0) / res)))
                c1 = min(nx, int(np.ceil((b[2] - x0) / res)))
                r0 = max(0, int(np.floor((y1 - b[3]) / res)))
                r1 = min(ny, int(np.ceil((y1 - b[1]) / res)))
                if c1 <= c0 or r1 <= r0:
                    continue
                tmp = np.full((r1 - r0, c1 - c0), np.nan, "float32")
                reproject(
                    src,
                    tmp,
                    src_transform=d.transform,
                    src_crs=d.crs,
                    dst_transform=from_origin(x0 + c0 * res, y1 - r0 * res, res, res),
                    dst_crs=crs,
                    src_nodata=np.nan,
                    dst_nodata=np.nan,
                    resampling=Resampling.nearest,
                )
            dst = arr[r0:r1, c0:c1]
            m = np.isfinite(tmp) & (np.isnan(dst) | ((dst == 0) & (tmp != 0)))
            dst[m] = tmp[m]
        arrays[f] = arr
    x = x0 + res / 2 + res * np.arange(nx)
    y = y1 - res / 2 - res * np.arange(ny)
    return GridSource(plan["key"], arrays, x, y, res, crs.to_wkt())


class Writer:
    """Writes the COGs of a study, then packs them into container files."""

    def __init__(self, out, container):
        self.out, self.container = out, container
        self.dir = os.path.join(out, "_cogs") if container else out
        os.makedirs(self.dir, exist_ok=True)
        self.chunks = []

    def add(self, name, data, crs, transform, descriptions):
        write_cog(os.path.join(self.dir, name), data, crs, transform, descriptions)
        return self._register(name)

    def _register(self, name):
        chunk = {"file": name}
        self.chunks.append(chunk)
        return chunk

    def finish(self):
        """Pack the COGs into data.bin (data_1.bin, ... beyond MAX_BIN), set offsets; return the files to upload."""
        if not self.container:
            return [c["file"] for c in self.chunks]
        bins, fh, off = [], None, 0
        for c in self.chunks:
            path = os.path.join(self.dir, c["file"])
            size = os.path.getsize(path)
            if fh is None or off + size > MAX_BIN:
                if fh:
                    fh.close()
                bins.append("data.bin" if not bins else f"data_{len(bins)}.bin")
                fh, off = open(os.path.join(self.out, bins[-1]), "wb"), 0
            pad = (-off) % ALIGN
            fh.write(b"\0" * pad)
            off += pad
            with open(path, "rb") as src:
                shutil.copyfileobj(src, fh, 1 << 24)
            c.update(bin=bins[-1], offset=off, size=size)
            off += size
        if fh:
            fh.close()
        shutil.rmtree(self.dir)
        return bins


def grid_of(ds, crs_txt, src):
    if not crs_txt:
        sys.exit(
            f"{src}: no CRS in the file (empty pyproj_srs); give it with --crs or 'crs:' in the study file"
        )
    crs = CRS.from_user_input(crs_txt)
    x, y, dx = ds.x, ds.y, ds.dx
    top = y[0] + dx / 2
    transform = from_origin(x[0] - dx / 2, top, dx, dx)
    bounds = [
        float(x[0] - dx / 2),
        float(top - len(y) * dx),
        float(x[-1] + dx / 2),
        float(top),
    ]
    return crs, transform, bounds, [len(y), len(x)]


def quantize(a, scale):
    q = np.empty(a.shape, "int16")
    for b, qb in zip(a, q):
        v = np.clip(np.round(b * scale), -32767, 32767)
        v[np.isnan(v)] = NODATA
        qb[:] = v
    return q


def choose_scale(a, info, tag=""):
    peak = (
        max(float(np.nanmax(a)), -float(np.nanmin(a))) if np.isfinite(a).any() else 0.0
    )
    scale = info["scale"]
    if scale is None:
        scale = 10 ** max(0, int(np.floor(np.log10(30000 / peak)))) if peak > 0 else 1
        scale = min(scale, 10000)
    if peak * scale > 32767:
        new = 10 ** int(np.floor(np.log10(32767 / peak))) if peak > 0 else 1
        print(f"  {tag}: exceeds int16 at scale {scale}, using scale {new}")
        scale = new
    return scale


def chunk_length(n, opt):
    if opt:
        return int(opt)
    return n if n <= 40 else 25


def export_region(region, runs, opts, writer, colour):
    """runs: list of (scenario, region config); scenario '' when the study has none.
    Returns the region description and, per scenario, its time axis and layers."""
    info_region, data, static = None, {}, {}
    for scen, cfg in runs:
        tag = f"{region}{'/' + scen if scen else ''}"
        ds = cfg["make_source"]() if "make_source" in cfg else Source(cfg["file"])
        times = ds.times + float(cfg.get("time_offset", 0))
        idx = time_indices(times, opts["time_steps"])
        sel_times = [float(times[i]) for i in idx]
        crs, transform, bounds, grid = grid_of(
            ds, cfg.get("crs") or ds.pyproj_srs or "", ds.name
        )
        if info_region is None:
            info_region = {
                "key": region,
                "label": cfg.get("label", region),
                "crs": f"EPSG:{crs.to_epsg()}" if crs.to_epsg() else f"IGM:{region}",
                "crs_proj4": crs.to_proj4(),
                "resolution_m": ds.dx,
                "bounds": bounds,
                "bounds_wgs84": list(transform_bounds(crs, "EPSG:4326", *bounds)),
                "grid": grid,
            }
        elif bounds != info_region["bounds"] or grid != info_region["grid"]:
            sys.exit(f"{tag}: the scenarios of a region must share their grid")
        prefix = f"{region}__{scen}__" if scen else f"{region}__"
        layers = {}
        thk = ds.read("thk", idx)
        for f in opts["layers"]:
            a = ds.read(f, idx)
            if a is None:
                print(f"  {tag}: {f} not in {ds.name}, skipped")
                continue
            if a.shape[0] > 1 and all(
                np.allclose(b, a[0], equal_nan=True) for b in a[1:]
            ):
                a = a[:1]
            info = layer_info(ds, f, opts["overrides"])
            colour.setdefault(f, {"info": info, "ranges": []})
            if "valid_min" in info:
                a[a < info["valid_min"]] = np.nan
            if "valid_max" in info:
                a[a > info["valid_max"]] = np.nan
            if info["mask"] == "ice" and thk is not None:
                if a.shape[0] > 1:
                    for b, tb in zip(a, thk[: a.shape[0]]):
                        b[~(tb > opts["ice_threshold"])] = np.nan
                else:
                    ever = np.zeros(thk.shape[1:], bool)
                    for tb in thk:
                        ever |= tb > opts["ice_threshold"]
                    a[0][~ever] = np.nan
            scale = choose_scale(a, info, f"{tag}: {f}")
            colour[f]["ranges"].append(
                colour_range(a[-1], info["colormap"], info["log"])
            )
            q = quantize(a, scale)
            del a
            nb = q.shape[0]
            entry = {"scale": scale, "nodata": NODATA, "static": nb == 1}
            key = (hashlib.sha1(q.tobytes()).hexdigest(), scale) if nb == 1 else None
            if key is not None and static.get(f, (None,))[0] == key:
                entry["chunks"] = static[f][1]["chunks"]
                layers[f] = entry
                if opts["world"] is not None:
                    opts["world"].setdefault((scen, f), []).append(
                        world_part(q[-1], scale, crs, transform)
                    )
                print(f"  {tag}: {f:12s}    1 band, shared")
                continue
            tc = chunk_length(nb, opts["time_chunk"])
            entry["chunks"] = []
            for i0 in range(0, nb, tc):
                part = q[i0 : i0 + tc]
                steps = sel_times[i0 : i0 + tc] if nb > 1 else sel_times[-1:]
                if nb > 1:
                    name = f"{prefix}{f}__{i0:04d}.tif"
                else:
                    name = (
                        f"{region}__{f}.tif" if f not in static else f"{prefix}{f}.tif"
                    )
                chunk = writer.add(
                    name, part, crs, transform, [f"TIME={t:g}" for t in steps]
                )
                chunk.update(t0=i0, n=int(part.shape[0]))
                entry["chunks"].append(chunk)
            if opts["world"] is not None:
                opts["world"].setdefault((scen, f), []).append(
                    world_part(q[-1], scale, crs, transform)
                )
            if key is not None and f not in static:
                static[f] = (key, entry)
            layers[f] = entry
            size = sum(
                os.path.getsize(os.path.join(writer.dir, c["file"]))
                for c in entry["chunks"]
            )
            print(
                f"  {tag}: {f:12s} {nb:4d} step(s) in {len(entry['chunks'])} chunk(s), "
                f"scale {scale:<6g} {size / 1e6:8.2f} MB"
            )
        ds.close()
        data[scen] = {"time": sel_times, "layers": layers}
    return info_region, data


VIEWER_URL = "https://igm-model.org/latest/viewer/app/"
VIEWER_FILES = [
    "index.html",
    "viewer.js",
    "viewer.css",
    "colormaps.json",
    "lib/ol.js",
    "lib/ol.css",
    "lib/geotiff.js",
    "lib/proj4.js",
    "view3d.js",
    "lib/three.module.min.js",
    "lib/OrbitControls.js",
    "lib/MapControls.js",
]


def viewer_copy(refresh=False):
    import urllib.request

    import time
    import urllib.error

    cache = os.path.join(os.path.expanduser("~"), ".cache", "igm", "viewer")
    index = os.path.join(cache, "index.html")
    if os.path.isfile(index) and time.time() - os.path.getmtime(index) > 86400:
        refresh = True
    for f in VIEWER_FILES:
        path = os.path.join(cache, f)
        if refresh or not os.path.isfile(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                urllib.request.urlretrieve(VIEWER_URL + f, path)
            except urllib.error.HTTPError as e:
                if f == "index.html":
                    raise
                print(f"viewer file {f} not available ({e.code}), skipped")
    return cache


def serve(folder, port, refresh=False, viewer=None):
    import re
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

    viewer = os.path.abspath(viewer) if viewer else viewer_copy(refresh)
    entry = json.load(open(os.path.join(folder, "studies_entry.json")))
    entry.update(base_url="..", url_suffix="")
    with open(os.path.join(folder, "studies_local.json"), "w") as fh:
        json.dump({"studies": [entry]}, fh, indent=1)

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=folder, **k)

        def translate_path(self, path):
            if path.startswith("/viewer/"):
                return os.path.join(viewer, path.split("?")[0][len("/viewer/") :])
            return super().translate_path(path)

        def end_headers(self):
            self.send_header("Accept-Ranges", "bytes")
            super().end_headers()

        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.translate_path(self.path)
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
            if not m or not os.path.isfile(path):
                return super().do_GET()
            size = os.path.getsize(path)
            a, b = m.groups()
            start = int(a) if a else max(size - int(b), 0)
            end = min(int(b), size - 1) if a and b else size - 1
            self.send_response(206)
            self.send_header("Content-Type", self.guess_type(path))
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as fh:
                fh.seek(start)
                self.wfile.write(fh.read(end - start + 1))

    url = f"http://127.0.0.1:{port}/viewer/index.html?catalogue=../studies_local.json"
    print(f"Serving {folder} (Ctrl+C to stop). Open:\n{url}")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    except KeyboardInterrupt:
        pass


def unpack(folder):
    """Write the COGs of a container export as plain files in <folder>/files."""
    meta = json.load(open(os.path.join(folder, "igm_viewer.json")))
    out = os.path.join(folder, "files")
    os.makedirs(out, exist_ok=True)
    seen = set()
    for scen in meta["data"].values():
        for reg in scen.values():
            for lay in reg["layers"].values():
                for c in lay["chunks"]:
                    if "bin" not in c or c["file"] in seen:
                        continue
                    seen.add(c["file"])
                    with open(os.path.join(folder, c["bin"]), "rb") as fh:
                        fh.seek(c["offset"])
                        with open(os.path.join(out, c["file"]), "wb") as dst:
                            dst.write(fh.read(c["size"]))
    print(f"{len(seen)} files written to {out}")


def study_id(cfg):
    """YEAR-FirstAuthor-JOURNAL, e.g. 2026-Doe-TC."""
    authors = cfg.get("authors") or ""
    first = authors[0] if isinstance(authors, list) and authors else str(authors)
    name = first.split(",")[0].split()[-1] if first.strip() else ""
    name = re.sub(r"[^A-Za-z]", "", unicodedata.normalize("NFKD", name))
    if cfg.get("year") and name and cfg.get("journal"):
        return f"{cfg['year']}-{name.capitalize()}-{cfg['journal']}"
    return None


def load_study(args):
    if args.input.endswith((".yaml", ".yml")):
        with open(args.input) as fh:
            cfg = yaml.safe_load(fh)
        base = os.path.dirname(os.path.abspath(args.input))
        if "tiles" in cfg:
            regions = {}
            for plan in plan_tiles(cfg["tiles"], base):
                regions[plan["key"]] = {
                    "make_source": (lambda plan=plan: build_tiles(plan)),
                    "label": plan["label"],
                    "crs": plan["crs"].to_wkt(),
                }
            cfg["regions"] = regions
            return cfg
        scen = cfg.get("scenarios") or {"": {"regions": cfg.get("regions") or {}}}
        for sc in scen.values():
            for r in sc["regions"].values():
                pats = r["file"] if isinstance(r["file"], list) else [r["file"]]
                files = []
                for pat in pats:
                    found = sorted(glob.glob(os.path.join(base, pat)))
                    if not found:
                        sys.exit(f"{args.input}: no file matches {pat}")
                    files += found
                r["file"] = files if len(files) > 1 else files[0]
        return cfg
    region = args.region or "region"
    return {
        "id": args.id,
        "title": args.title or region,
        "regions": {region: {"file": args.input}},
    }


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="igm_export_for_viewer",
        description="Export IGM netCDF outputs as a study for the IGM map viewer.",
    )
    p.add_argument(
        "input", nargs="?", help="study file (.yaml) or a single netCDF output (.nc)"
    )
    p.add_argument(
        "--out", default="viewer_export", help="output folder (default: viewer_export)"
    )
    p.add_argument(
        "--crs", help="CRS of the model grid, e.g. EPSG:32632, when the netCDF has none"
    )
    p.add_argument("--region", help="region name (single netCDF input)")
    p.add_argument("--id", help="study id YEAR-FirstAuthor-JOURNAL, e.g. 2026-Doe-TC")
    p.add_argument("--title", help="study title (single netCDF input)")
    p.add_argument(
        "--layers",
        help="comma-separated layers (default: the usual IGM fields found in the file)",
    )
    p.add_argument(
        "--time-steps", help="'all' (default), 'every:N', or comma-separated times"
    )
    p.add_argument(
        "--time-chunk",
        type=int,
        help="time steps per file (default: all up to 40 steps, else 25)",
    )
    p.add_argument(
        "--no-container",
        action="store_true",
        help="keep one file per chunk instead of packing them into data.bin",
    )
    p.add_argument(
        "--force", action="store_true", help="overwrite a non-empty output folder"
    )
    p.add_argument(
        "--serve",
        metavar="DIR",
        help="serve an exported folder locally and print the viewer URL to check it",
    )
    p.add_argument(
        "--port", type=int, default=8765, help="port for --serve (default: 8765)"
    )
    p.add_argument(
        "--viewer", metavar="DIR", help="with --serve, use this copy of the viewer"
    )
    p.add_argument(
        "--refresh-viewer",
        action="store_true",
        help="with --serve, download the viewer again",
    )
    p.add_argument(
        "--unpack",
        metavar="DIR",
        help="write the files packed in DIR/data.bin as plain COGs in DIR/files",
    )
    args = p.parse_args(argv)
    if args.serve:
        return serve(args.serve, args.port, args.refresh_viewer, args.viewer)
    if args.unpack:
        return unpack(args.unpack)
    if not args.input:
        p.error("give a study file or a netCDF output")

    cfg = load_study(args)
    if os.path.isdir(args.out) and os.listdir(args.out):
        if not args.force:
            sys.exit(f"{args.out} is not empty (use --force to overwrite)")
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    try:
        from importlib.metadata import version

        igm_version = version("igm-model")
    except Exception:
        igm_version = ""
    cfg["id"] = cfg.get("id") or study_id(cfg)
    if not cfg["id"]:
        sys.exit("no study id: give 'id', or 'year', 'authors' and 'journal'")
    if not re.fullmatch(r"\d{4}-[A-Z][A-Za-z]*-[A-Za-z0-9]+", cfg["id"]):
        print(f"WARNING: id {cfg['id']} does not follow YEAR-FirstAuthor-JOURNAL")
    if isinstance(cfg.get("authors"), list):
        cfg["authors"] = ", ".join(cfg["authors"])
    keys = ("id", "title", "authors", "year", "journal", "doi", "paper", "licence")
    study = {k: cfg[k] for k in keys if cfg.get(k) not in (None, "")}
    study.setdefault("licence", "CC-BY-4.0")
    study["igm_version"] = str(cfg.get("igm_version", igm_version))

    layers = (
        args.layers.split(",") if args.layers else cfg.get("layers", DEFAULT_LAYERS)
    )
    time_steps = cfg.get("time_steps", "all")
    if args.time_steps:
        ts = args.time_steps
        time_steps = (
            ts
            if ts == "all" or ts.startswith("every:")
            else [float(t) for t in ts.split(",")]
        )
    scenarios = cfg.get("scenarios") or {"": {"regions": cfg["regions"]}}
    regions = list(next(iter(scenarios.values()))["regions"])
    for sname, sc in scenarios.items():
        if sname and not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", sname):
            sys.exit(f"scenario {sname}: use lowercase letters, digits, - or _")
        if sorted(sc["regions"]) != sorted(regions):
            sys.exit(f"scenario {sname}: every scenario must have the same regions")
    for r in regions:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", r):
            sys.exit(f"region {r}: use lowercase letters, digits, - or _")
    inherit = {k: cfg[k] for k in ("crs", "time_offset") if k in cfg}
    if args.crs:
        inherit.setdefault("crs", args.crs)
    opts = dict(
        layers=layers,
        overrides=cfg.get("layer_options", {}),
        world={} if cfg.get("overview", cfg.get("preview", len(regions) > 1)) else None,
        time_steps=time_steps,
        time_chunk=args.time_chunk or cfg.get("time_chunk"),
        ice_threshold=float(cfg.get("ice_threshold", 1.0)),
    )
    writer = Writer(args.out, not args.no_container)
    colour, region_list, data = {}, [], {s: {} for s in scenarios}
    for region in regions:
        runs = []
        for sname, sc in scenarios.items():
            rcfg = dict(inherit)
            rcfg.update({k: sc[k] for k in ("crs", "time_offset") if k in sc})
            rcfg.update(sc["regions"][region])
            runs.append((sname, rcfg))
            fl = rcfg.get("file", [])
            fl = fl if isinstance(fl, list) else [fl]
            print(
                f"{region}{'/' + sname if sname else ''}: "
                + (", ".join(os.path.basename(f) for f in fl) or rcfg.get("label", ""))
            )
        info_region, per_scen = export_region(region, runs, opts, writer, colour)
        region_list.append(info_region)
        for sname, d in per_scen.items():
            data[sname][region] = d

    if opts["world"]:
        wb, winfo = None, None
        for (sname, f), parts in opts["world"].items():
            arr, tr, res = assemble_world(parts)
            info = colour[f]["info"]
            scale = choose_scale(arr, info, f"world: {f}")
            t_last = data[sname][regions[0]]["time"][-1]
            chunk = writer.add(
                f"world__{sname + '__' if sname else ''}{f}.tif",
                quantize(arr[None], scale),
                "EPSG:3857",
                tr,
                [f"TIME={t_last:g}"],
            )
            chunk.update(t0=0, n=1)
            d = data[sname].setdefault("world", {"time": [t_last], "layers": {}})
            d["layers"][f] = {
                "scale": scale,
                "nodata": NODATA,
                "static": True,
                "chunks": [chunk],
            }
            b = list(_bounds(tr, arr.shape[1], arr.shape[0]))
            wb = (
                b
                if wb is None
                else [
                    min(wb[0], b[0]),
                    min(wb[1], b[1]),
                    max(wb[2], b[2]),
                    max(wb[3], b[3]),
                ]
            )
            winfo = (res, arr.shape)
            print(
                f"  world: {f:12s} {arr.shape[1]} x {arr.shape[0]} at {res / 1000:g} km"
            )
        region_list.insert(
            0,
            {
                "key": "world",
                "label": "World overview",
                "overview": True,
                "crs": "EPSG:3857",
                "crs_proj4": CRS.from_epsg(3857).to_proj4(),
                "resolution_m": winfo[0],
                "bounds": wb,
                "bounds_wgs84": list(transform_bounds("EPSG:3857", "EPSG:4326", *wb)),
                "grid": [
                    int(round((wb[3] - wb[1]) / winfo[0])),
                    int(round((wb[2] - wb[0]) / winfo[0])),
                ],
            },
        )

    layer_meta = {}
    for f in layers:
        if f not in colour:
            continue
        info, ranges = colour[f]["info"], colour[f]["ranges"]
        layer_meta[f] = {
            "label": info["label"],
            "unit": info["unit"],
            "colormap": info["colormap"],
            "log": bool(info["log"]),
            "mask": info["mask"],
            "vmin": info.get(
                "vmin",
                (
                    min(r[0] for r in ranges)
                    if len(ranges) <= 6
                    else nice(float(np.median([r[0] for r in ranges])))
                ),
            ),
            "vmax": info.get(
                "vmax",
                (
                    max(r[1] for r in ranges)
                    if len(ranges) <= 6
                    else nice(float(np.median([r[1] for r in ranges])))
                ),
            ),
        }
    uploads = writer.finish()
    meta = {
        "schema": "igm_viewer/2",
        "study": study,
        "time_unit": "yr",
        "time_label": "year",
        "blocksize": BLOCK,
        "scenarios": [
            {"key": s, "label": sc.get("label", s)} for s, sc in scenarios.items()
        ],
        "regions": region_list,
        "layers": layer_meta,
        "data": data,
    }
    with open(os.path.join(args.out, "igm_viewer.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    uploads = ["igm_viewer.json"] + uploads
    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in uploads)
    print(f"\n{len(uploads)} files to upload, {total / 1e6:.1f} MB in {args.out}")
    if len(uploads) > 100:
        print("WARNING: a Zenodo record holds at most 100 files")
    if total > 50e9:
        print("WARNING: a Zenodo record holds at most 50 GB")

    entry = {
        "id": study.get("id"),
        "title": study.get("title"),
        "base_url": "https://zenodo.org/api/records/<record>/files",
        "url_suffix": "/content",
    }
    if study.get("doi"):
        entry["doi"] = study["doi"]
    with open(os.path.join(args.out, "studies_entry.json"), "w") as fh:
        json.dump(entry, fh, indent=1)
    print(
        "\nEntry for the viewer catalogue (docs/viewer/app/studies.json), also in studies_entry.json:"
    )
    print(json.dumps(entry, indent=1))
    print(
        f"\nUpload {', '.join(uploads) if len(uploads) < 6 else 'the files listed above'} "
        "to a Zenodo record, then replace <record> by its number."
    )
    print(f"Check it first with: igm_export_for_viewer --serve {args.out}")


if __name__ == "__main__":
    main()
