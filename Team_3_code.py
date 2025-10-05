#!/usr/bin/env python3
"""
Operator node simulation for Team 3:
 - Propagate TLE (sgp4).
 - Add random sensor bias + gaussian noise.
 - *** NEW: Implement a simple Kalman Filter (EKF) to produce a local orbit estimate from noisy measurements. ***
 - Compute grid-cell risk from the filtered estimate.
 - Anonymize outputs and save JSON.
"""

import json
import hashlib
import numpy as np
import pandas as pd
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
from astropy.time import Time
from astropy import units as u
from astropy.coordinates import TEME, ITRS, CartesianRepresentation
from astropy.coordinates import EarthLocation

# -------------------------
# Configuration for Node 3
# -------------------------
CONFIG = {
    "node_name": "operator_node_gamma",
    "tle": [
        "1 43226U 18017A   25305.54583333  .00001234  00000-0  24691-4 0  9993",
        "2 43226  97.4442 211.2323 0008544 263.6345  96.4433 15.24453333 98765"
    ],
    "start_utc": "2025-10-03T00:00:00Z",
    "duration_minutes": 120,
    "step_seconds": 60,
    "sensor": {
        "position_bias_m": [-5.0, 12.0, -8.0],
        "position_noise_std_m": 30.0,
    },
    "grid": {
        "lat_step_deg": 0.5,
        "lon_step_deg": 0.5,
        "extent_km": 2000,
        "risk_kernel_sigma_km": 200.0
    },
    "anonymize_salt": "a_completely_different_salt_for_gamma",
    "max_grid_cells_returned": 200
}

# -----------------------------------
# NEW: Kalman Filter Implementation
# -----------------------------------
class SimpleEKF:
    def __init__(self, initial_state, initial_covariance, process_noise_std, measurement_noise_std):
        self.x = initial_state
        self.P = initial_covariance
        self.Q = np.diag([process_noise_std**2] * 6)
        self.R = np.diag([measurement_noise_std**2] * 3)
        self.H = np.zeros((3, 6))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1

    def predict(self, dt):
        F = np.identity(6)
        F[0, 3] = dt; F[1, 4] = dt; F[2, 5] = dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement_xyz):
        y = measurement_xyz - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.identity(6) - K @ self.H) @ self.P

# -------------------------
# Helpers
# -------------------------
def anonymize_id(raw: str, salt: str) -> str:
    h = hashlib.sha256((salt + raw).encode("utf-8")).hexdigest()
    return "sat-" + h[:12]

def propagate_tle_to_states(tle_lines, start_utc_iso, duration_minutes, step_seconds):
    sat = Satrec.twoline2rv(tle_lines[0], tle_lines[1])
    t0 = datetime.fromisoformat(start_utc_iso.replace("Z", "+00:00"))
    n_steps = int((duration_minutes * 60) / step_seconds) + 1
    states = []
    for i in range(n_steps):
        t = t0 + timedelta(seconds=i * step_seconds)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond*1e-6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0: raise RuntimeError(f"SGP4 propagation error code {e}")
        states.append({"utc": t.isoformat() + "Z", "r_teme_km": np.array(r), "v_teme_km_s": np.array(v)})
    return states

def teme_to_latlonalt(state_r_km, utc_time):
    t = Time(utc_time)
    r = CartesianRepresentation(state_r_km * u.km)
    teme_coord = TEME(r, obstime=t)
    itrs = teme_coord.transform_to(ITRS(obstime=t))
    el = EarthLocation(x=itrs.x, y=itrs.y, z=itrs.z)
    return el.lat.deg, el.lon.deg, el.height.to(u.m).value

def add_sensor_noise_and_bias(r_km, config_sensor):
    r_m = np.array(r_km) * 1000.0
    bias = np.array(config_sensor["position_bias_m"])
    noise = np.random.normal(0.0, config_sensor["position_noise_std_m"], size=3)
    measured = r_m + bias + noise
    cov = np.diag([config_sensor["position_noise_std_m"]**2]*3)
    return measured, cov

def cov_to_uncertainty_ellipse_2d(cov_3x3):
    cov2 = cov_3x3[:2, :2]
    vals, vecs = np.linalg.eigh(cov2)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    semi_major = np.sqrt(vals[0]) * 2.4477
    semi_minor = np.sqrt(vals[1]) * 2.4477
    angle_rad = np.arctan2(vecs[1, 0], vecs[0, 0])
    return float(semi_major), float(semi_minor), np.degrees(angle_rad)

def build_grid(lat_step, lon_step):
    lats = np.arange(-90, 90 + 1e-6, lat_step)
    lons = np.arange(-180, 180 + 1e-6, lon_step)
    latg, longg = np.meshgrid(lats, lons, indexing='ij')
    return pd.DataFrame({"lat": latg.ravel(), "lon": longg.ravel()})

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1r, lat2r, dlon = map(np.radians, [lat1, lat2, lon2 - lon1])
    dlat = lat2r - lat1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def compute_grid_risk(subsat_points, grid_df, kernel_sigma_km, extent_km):
    risk = np.zeros(len(grid_df))
    cell_lats, cell_lons = grid_df['lat'].values, grid_df['lon'].values
    for latc, lonc in subsat_points:
        dists = haversine_km(latc, lonc, cell_lats, cell_lons)
        mask = dists <= extent_km
        if not np.any(mask): continue
        kernel_vals = np.exp(-0.5 * (dists[mask] / kernel_sigma_km)**2)
        risk[mask] += kernel_vals
    if np.max(risk) > 0: risk /= np.max(risk)
    grid_df['risk'] = risk.astype(float)
    return grid_df

# -------------------------
# Main simulation routine
# -------------------------
def run_simulation(config):
    raw_states = propagate_tle_to_states(config["tle"], config["start_utc"], config["duration_minutes"], config["step_seconds"])
    first_state = raw_states[0]
    initial_measured_pos, initial_cov = add_sensor_noise_and_bias(first_state['r_teme_km'], config["sensor"])
    initial_vel_ms = first_state['v_teme_km_s'] * 1000.0
    initial_state_vec = np.hstack([initial_measured_pos, initial_vel_ms])
    initial_P = np.diag([config["sensor"]["position_noise_std_m"]**2]*3 + [100**2]*3)

    ekf = SimpleEKF(initial_state=initial_state_vec, initial_covariance=initial_P,
                    process_noise_std=1.0, measurement_noise_std=config["sensor"]["position_noise_std_m"])

    out_states = []
    subsat_list_filtered = []

    for i, st in enumerate(raw_states):
        utc = st['utc']
        r_teme_km, v_teme_km_s = st['r_teme_km'], st['v_teme_km_s']
        measured_pos_m, measurement_cov_m2 = add_sensor_noise_and_bias(r_teme_km, config["sensor"])
        if i > 0:
            dt = config["step_seconds"]
            ekf.predict(dt)
        ekf.update(measured_pos_m)
        filtered_pos_m, filtered_vel_ms, filtered_cov_m2 = ekf.x[:3], ekf.x[3:], ekf.P
        try:
            lat, lon, alt_m = teme_to_latlonalt(filtered_pos_m / 1000.0, utc)
            if not np.isnan(lat):
                subsat_list_filtered.append((lat, lon))
        except Exception:
            lat, lon, alt_m = float('nan'), float('nan'), float('nan')
        semi_major_m, semi_minor_m, angle_deg = cov_to_uncertainty_ellipse_2d(filtered_cov_m2)
        out_states.append({
            "utc": utc,
            "ground_truth_teme_km": {"r": r_teme_km.tolist(), "v": v_teme_km_s.tolist()},
            "noisy_measurement": {"r_m": measured_pos_m.tolist(), "cov_m2": measurement_cov_m2.tolist()},
            "filtered_state_estimate": {
                "r_teme_m": filtered_pos_m.tolist(),
                "v_teme_ms": filtered_vel_ms.tolist(),
                "cov_m2": filtered_cov_m2.tolist(),
                "subsat_lat_deg": lat, "subsat_lon_deg": lon, "subsat_alt_m": alt_m,
                "uncertainty_ellipse_m": {"semi_major_m": semi_major_m, "semi_minor_m": semi_minor_m, "orientation_deg": angle_deg}
            }
        })

    grid = build_grid(config["grid"]["lat_step_deg"], config["grid"]["lon_step_deg"])
    grid_with_risk = compute_grid_risk(subsat_list_filtered, grid, config["grid"]["risk_kernel_sigma_km"], config["grid"]["extent_km"])
    top_cells = grid_with_risk.sort_values("risk", ascending=False).head(config["max_grid_cells_returned"])

    anon_sat_id = anonymize_id(config["tle"][1], config["anonymize_salt"])
    anon_node_id = anonymize_id(config["node_name"], config["anonymize_salt"])

    summary = {
        "node_id": anon_node_id, "sat_id": anon_sat_id,
        "start_utc": config["start_utc"], "duration_minutes": config["duration_minutes"],
        "states": out_states,
        "grid_cells": top_cells.to_dict(orient='records'),
    }
    return summary

# -------------------------
# Execute and write JSON
# -------------------------
if __name__ == "__main__":
    np.random.seed(1337)
    summary = run_simulation(CONFIG)
    out_file = "anonymized_operator_node_3_summary.json"
    with open(out_file, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Wrote anonymized JSON summary for Node 3 -> {out_file}")
