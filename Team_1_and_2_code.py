#!/usr/bin/env python3
"""
Operator node simulation:

 - propagate TLE (sgp4)
 - convert TEME -> ITRS -> lat/lon/alt (astropy)
 - add random sensor bias + gaussian noise
 - compute uncertainty ellipse (2D) from covariance estimate
 - compute grid-cell risk (simple gaussian kernel on ground cells)
 - anonymize outputs and save JSON
"""

import json
import hashlib
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, CartesianRepresentation, EarthLocation
from scipy.spatial.distance import cdist

# -------------------------
# Configuration
# -------------------------
CONFIG = {
    "node_name": "operator_node_alpha",   # will be anonymized in final JSON
    "tle": [
        "1 25544U 98067A   25304.90585560  .00002182  00000+0  47424-4 0  9990",
        "2 25544  51.6444  12.3456 0009896 123.4567 236.5433 15.50000000    12"
    ],
    "start_utc": "2025-10-03T00:00:00Z",   # ISO-8601 UTC
    "duration_minutes": 120,
    "step_seconds": 60,
    "sensor": {
        "position_bias_m": [10.0, -7.0, 5.0],      # constant bias (m) applied to x,y,z
        "position_noise_std_m": 25.0,              # Gaussian noise std dev (m)
        "attitude_noise_deg": 0.5,                 # example
    },
    "grid": {
        "lat_step_deg": 0.5,
        "lon_step_deg": 0.5,
        "extent_km": 2000,    # kernel extent for risk aggregation
        "risk_kernel_sigma_km": 200.0
    },
    "anonymize_salt": "replace_with_operator_provided_salt",
    "max_grid_cells_returned": 200
}

# -------------------------
# Helpers
# -------------------------
def anonymize_id(raw: str, salt: str) -> str:
    """Return a short anonymized id for any raw string using SHA256."""
    h = hashlib.sha256((salt + raw).encode("utf-8")).hexdigest()
    return "sat-" + h[:12]


def propagate_tle_to_states(tle_lines, start_utc_iso, duration_minutes, step_seconds):
    """Propagate TLE using sgp4 and return list of state dicts: time, r_teme_km, v_teme_km_s."""
    sat = Satrec.twoline2rv(tle_lines[0], tle_lines[1])
    t0 = datetime.fromisoformat(start_utc_iso.replace("Z", "+00:00"))
    n_steps = int((duration_minutes * 60) / step_seconds) + 1
    states = []
    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            # sgp4 returns error code in e; handle gracefully
            raise RuntimeError(f"SGP4 propagation error code {e} at time {t.isoformat()}")
        # r, v are in TEME frame km and km/s
        states.append({"utc": t.isoformat() + "Z", "r_teme_km": np.array(r), "v_teme_km_s": np.array(v)})
    return states

def teme_to_latlonalt(state_r_km, utc_time):
    """Convert TEME position vector (km) at utc_time -> lat, lon, alt using astropy."""
    t = Time(utc_time)  # expects ISO
    # Create TEME cartesian
    r = CartesianRepresentation(state_r_km * u.km)
    teme_coord = TEME(r, obstime=t)
    # Transform to ITRS (ECEF) then to lat/lon/height (geodetic)
    itrs = teme_coord.transform_to(ITRS(obstime=t))
    # Access geodetic lat lon alt via EarthLocation
    x = itrs.x.to(u.m).value
    y = itrs.y.to(u.m).value
    z = itrs.z.to(u.m).value
    el = EarthLocation(x=x*u.m, y=y*u.m, z=z*u.m)
    lat = el.lat.deg
    lon = el.lon.deg
    alt_m = el.height.to(u.m).value
    return lat, lon, alt_m

def add_sensor_noise_and_bias(r_km, config_sensor):
    """Simulate sensor: add constant bias and gaussian noise. Returns noisy measurement in meters."""
    r_m = np.array(r_km) * 1000.0
    bias = np.array(config_sensor["position_bias_m"])
    noise = np.random.normal(0.0, config_sensor["position_noise_std_m"], size=3)
    measured = r_m + bias + noise
    # Simple covariance estimate: diag(noise_std^2) in meters^2
    cov = np.diag([config_sensor["position_noise_std_m"]**2]*3)
    return measured, cov

def cov_to_uncertainty_ellipse_2d(cov_3x3):
    """
    Create a 2D uncertainty ellipse (east,north) from the 3x3 covariance in ECEF m^2.
    We'll project to a local ENU plane by taking the top-left 2x2 of the covariance as a simplification.
    Returns semi-major (m), semi-minor (m), angle_deg (east->north).
    """
    cov2 = cov_3x3[:2, :2]
    vals, vecs = np.linalg.eigh(cov2)
    # sort descending
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    semi_major = np.sqrt(vals[0]) * 2.4477  # 95% ~ chi2 factor ~2.4477 (sqrt of chi2(2,0.95))
    semi_minor = np.sqrt(vals[1]) * 2.4477
    # angle of major axis wrt east axis
    angle_rad = np.arctan2(vecs[1, 0], vecs[0, 0])
    angle_deg = np.degrees(angle_rad)
    return float(semi_major), float(semi_minor), float(angle_deg)

def build_grid(lat_min=-90, lat_max=90, lon_min=-180, lon_max=180, lat_step=0.5, lon_step=0.5):
    lats = np.arange(lat_min, lat_max + 1e-6, lat_step)
    lons = np.arange(lon_min, lon_max + 1e-6, lon_step)
    latg, longg = np.meshgrid(lats, lons, indexing='ij')
    flat = pd.DataFrame({
        "lat": latg.ravel(),
        "lon": longg.ravel()
    })
    return flat

def haversine_km(lat1, lon1, lat2, lon2):
    """Great circle distance in km between two arrays (vectorized)."""
    R = 6371.0
    lat1r = np.radians(lat1)
    lat2r = np.radians(lat2)
    dlat = lat2r - lat1r
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def compute_grid_risk(subsat_points, grid_df, kernel_sigma_km=200.0, extent_km=2000.0):
    """
    For each subsatellite point (lat,lon), build/add a gaussian kernel of risk to grid cells.
    subsat_points: list of (lat, lon) tuples
    grid_df: DataFrame with 'lat','lon'
    Returns a DataFrame with an added 'risk' column.
    """
    risk = np.zeros(len(grid_df))
    # Precompute cell centers arrays
    cell_lats = grid_df['lat'].values
    cell_lons = grid_df['lon'].values
    # For efficiency, only consider subset within extent_km of each point
    for latc, lonc in subsat_points:
        dists = haversine_km(latc, lonc, cell_lats, cell_lons)
        mask = dists <= extent_km
        if not np.any(mask):
            continue
        # gaussian kernel
        kernel_vals = np.exp(-0.5 * (dists[mask] / kernel_sigma_km)**2)
        risk[mask] += kernel_vals
    # Normalize risk to 0-1
    if np.max(risk) > 0:
        risk = risk / np.max(risk)
    grid_df = grid_df.copy()
    grid_df['risk'] = (risk).astype(float)
    return grid_df

# -------------------------
# Main simulation routine
# -------------------------
def run_simulation(config):
    # 1) propagate
    raw_states = propagate_tle_to_states(
        config["tle"], config["start_utc"], config["duration_minutes"], config["step_seconds"]
    )

    # 2) for each state compute lat/lon and sensor measurements
    out_states = []
    subsat_list = []
    for st in raw_states:
        utc = st['utc']
        r_teme_km = st['r_teme_km']
        v_teme_km_s = st['v_teme_km_s']
        try:
            lat, lon, alt_m = teme_to_latlonalt(r_teme_km, utc)
        except Exception:
            # fallback: mark NaN but continue
            lat, lon, alt_m = float('nan'), float('nan'), float('nan')

        measured_m, cov_m = add_sensor_noise_and_bias(r_teme_km, config["sensor"])
        # Compute simplified uncertainty ellipse (projected)
        semi_major_m, semi_minor_m, angle_deg = cov_to_uncertainty_ellipse_2d(cov_m)

        out_states.append({
            "utc": utc,
            "r_teme_km": r_teme_km.tolist(),
            "v_teme_km_s": v_teme_km_s.tolist(),
            "subsat_lat_deg": float(lat),
            "subsat_lon_deg": float(lon),
            "subsat_alt_m": float(alt_m),
            "measured_position_m": measured_m.tolist(),
            "measurement_cov_m2": cov_m.tolist(),
            "uncertainty_ellipse_m": {
                "semi_major_m": semi_major_m,
                "semi_minor_m": semi_minor_m,
                "orientation_deg": angle_deg
            }
        })
        if not np.isnan(lat):
            subsat_list.append((lat, lon))

    # 3) grid risk
    grid = build_grid(lat_step=config["grid"]["lat_step_deg"], lon_step=config["grid"]["lon_step_deg"])
    grid_with_risk = compute_grid_risk(subsat_list, grid, kernel_sigma_km=config["grid"]["risk_kernel_sigma_km"], extent_km=config["grid"]["extent_km"])
    # We will keep only top N grid cells by risk to avoid huge JSON
    top_cells = grid_with_risk.sort_values("risk", ascending=False).head(config["max_grid_cells_returned"])
    grid_cells_out = top_cells.to_dict(orient='records')

    # 4) anonymize metadata
    raw_id = config.get("tle", [""])[1]  # use second line (contains NORAD id) as raw string to anonymize
    anon_sat_id = anonymize_id(raw_id, config["anonymize_salt"])
    anon_node_id = anonymize_id(config.get("node_name", "node"), config["anonymize_salt"])

    summary = {
        "node_id": anon_node_id,
        "sat_id": anon_sat_id,
        "start_utc": config["start_utc"],
        "duration_minutes": config["duration_minutes"],
        "step_seconds": config["step_seconds"],
        "states": out_states,
        "grid_cells": grid_cells_out,
        "metadata": {
            "sensor_model": {
                "position_bias_m": None,   # explicitly remove operator actual bias to preserve anonymization
                "position_noise_std_m": config["sensor"]["position_noise_std_m"]
            },
            "grid": {
                "lat_step_deg": config["grid"]["lat_step_deg"],
                "lon_step_deg": config["grid"]["lon_step_deg"]
            }
        }
    }
    return summary

# -------------------------
# Execute and write JSON
# -------------------------
if __name__ == "__main__":
    np.random.seed(42)  # deterministic example
    summary = run_simulation(CONFIG)
    out_file = "anonymized_operator_node_summary.json"
    with open(out_file, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote anonymized JSON summary -> {out_file}")
