import tensorflow as tf
import netCDF4
import numpy as np


def load_geology(path_nc):
    """
    Load geological data from a NetCDF file.

    Parameters:
    -----------
    path_nc : str
        Path to the NetCDF file

    Returns:
    --------
    topo : tf.Tensor
        Bedrock topography
    thk1880 : tf.Tensor
        Initial ice thickness (typically from 1880 or similar reference year)
    mask : tf.Tensor
        Ice mask
    """
    nc = netCDF4.Dataset(path_nc)
    topo = tf.constant(nc.variables['topg'][:], dtype=tf.float32)
    # Use surf_1999 - topo as initial thickness (can be changed based on available data)
    thk1880 = tf.constant(nc.variables['surf_1999'][:], dtype=tf.float32) - topo
    mask = tf.constant(nc.variables['icemask'][:], dtype=tf.float32)
    nc.close()
    return topo, thk1880, mask


def load_daily_data(file_path, accumulate=False):
    """
    Parse a whitespace-delimited file with columns:
        year  jd  hour  temp  prec
    (first two lines are headers), and pack into a tensor of shape:
        (num_years, 366, 2), where [:, :, 0] = temp, [:, :, 1] = prec

    Parameters
    ----------
    file_path : str
        Path to the input file (e.g., "temp_prec.dat").
    accumulate : bool, optional
        If multiple rows map to the same (year, day), whether to sum them (True)
        or let the last occurrence overwrite (False). Default False.

    Returns
    -------
    data : tf.Tensor
        Shape (num_years, 366, 2): temperature and precipitation grids.
    years : tf.Tensor
        Shape (num_years,): sorted unique calendar years aligned with data[year_idx].
    """
    # Read & parse file
    years_f, jds_f, temps, precs = [], [], [], []

    with open(file_path, "r") as f:
        lines = f.read().strip().splitlines()

    # Skip first 2 header rows
    for line in lines[2:]:
        if not line.strip():
            continue
        cols = line.split()
        if len(cols) < 5:
            continue
        y, jd, hr, te, pr = cols[:5]
        years_f.append(float(y))
        jds_f.append(float(jd))
        temps.append(float(te))
        precs.append(float(pr))

    if not years_f:
        # No data rows parsed; return empty tensors
        empty = tf.zeros((0, 366, 2), dtype=tf.float32)
        return empty, tf.zeros((0,), dtype=tf.int64)

    # Convert to numpy arrays first for processing
    year = np.array(years_f, dtype=np.float32)
    jd = np.array(jds_f, dtype=np.float32)
    temp = np.array(temps, dtype=np.float32)
    prec = np.array(precs, dtype=np.float32)

    # Cast year & jd to int
    year_i = year.astype(np.int64)
    jd_i = jd.astype(np.int64)

    # Sort by (year, jd)
    sort_keys = year_i * 1000 + jd_i
    order = np.argsort(sort_keys)
    year_i = year_i[order]
    jd_i = jd_i[order]
    temp = temp[order]
    prec = prec[order]

    # Unique sorted years
    years = np.unique(year_i)
    years.sort()

    # Create year index mapping
    year_to_idx = {y: i for i, y in enumerate(years)}
    yidx = np.array([year_to_idx[y] for y in year_i])

    # Day index: clamp to 1..366, then make 0-based
    didx = np.clip(jd_i, 1, 366) - 1

    num_years = len(years)

    # Initialize grids
    temp_grid = np.zeros((num_years, 366), dtype=np.float32)
    prec_grid = np.zeros((num_years, 366), dtype=np.float32)

    # Populate grids
    if accumulate:
        for i, (yi, di) in enumerate(zip(yidx, didx)):
            temp_grid[yi, di] += temp[i]
            prec_grid[yi, di] += prec[i]
    else:
        for i, (yi, di) in enumerate(zip(yidx, didx)):
            temp_grid[yi, di] = temp[i]
            prec_grid[yi, di] = prec[i]

    # Stack to (num_years, 366, 2)
    data = np.stack((temp_grid, prec_grid), axis=-1)

    return tf.constant(data, dtype=tf.float32), tf.constant(years, dtype=tf.int64)


def load_smb_point_obs(path_csv, alt_col="Alt", smb_col="Annual_SMB (m w.e.)",
                       to_ice_eq=True, rho_ice=0.917, with_year=False,
                       year_col="Year"):
    """
    Load stake SMB observations from a CSV.

    Expected columns: altitude and annual SMB (m w.e./yr). X/Y are ignored.
    Returns (altitudes, smb_values) as 1D numpy float32 arrays, with SMB
    converted to m ice eq./yr by default (dividing by rho_ice) so it plots
    on the same axis as the model's smb_vec.

    If with_year=True, also returns the observation year as a third array
    (read from year_col). When the column is absent the year array is None.
    """
    import csv
    alts, smbs, years = [], [], []
    have_year = False
    with open(path_csv, "r") as f:
        reader = csv.DictReader(f)
        have_year = year_col in (reader.fieldnames or [])
        for row in reader:
            try:
                alt = float(row[alt_col])
                smb = float(row[smb_col])
                year = float(row[year_col]) if have_year else np.nan
            except (KeyError, ValueError):
                continue
            alts.append(alt)
            smbs.append(smb)
            years.append(year)
    alts = np.asarray(alts, dtype=np.float32)
    smbs = np.asarray(smbs, dtype=np.float32)
    if to_ice_eq:
        smbs = smbs / rho_ice
    if with_year:
        years = np.asarray(years, dtype=np.float32) if have_year else None
        return alts, smbs, years
    return alts, smbs


def load_observations_from_nc(path_nc, topo, years=['surf_1880', 'surf_1926', 'surf_1957',
                                                    'surf_1980', 'surf_1999', 'surf_2009', 'surf_2017']):
    """
    Load observation surfaces from a NetCDF file.

    Parameters:
    -----------
    path_nc : str
        Path to the NetCDF file
    topo : tf.Tensor
        Bedrock topography to subtract from surfaces
    years : list
        List of variable names to load

    Returns:
    --------
    observations : dict
        Dictionary of ice thickness observations
    """
    nc = netCDF4.Dataset(path_nc)
    observations = {}
    for year in years:
        if year in nc.variables:
            surf = tf.constant(nc.variables[year][:], dtype=tf.float32)
            observations[year] = surf - topo
        else:
            print(f"Warning: {year} not found in {path_nc}")
    nc.close()
    return observations
