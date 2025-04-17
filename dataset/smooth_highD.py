
import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from tqdm import tqdm

# Configuration
RAW_PATH = "dataset/data/raw/highD/"
SMOOTHED_PATH = "dataset/data/smooth/highD/20hz/"
os.makedirs(SMOOTHED_PATH, exist_ok=True)

# Constants for smoothing
SMOOTH_WINDOW = 21  # must be odd
POLY_ORDER = 3

def savgol_smooth(arr, window=SMOOTH_WINDOW, poly=POLY_ORDER):
    if len(arr) < window:
        window = len(arr) if len(arr) % 2 == 1 else len(arr) - 1
    if window < 3:
        return arr  # not enough data to smooth
    return savgol_filter(arr, window_length=window, polyorder=min(poly, window - 1))

def compute_derivatives(df):
    dt = 0.04  # 25Hz original sampling rate → 40ms
    dx = np.gradient(df['Local_X'], dt)
    dy = np.gradient(df['Local_Y'], dt)
    ddx = np.gradient(dx, dt)
    ddy = np.gradient(dy, dt)
    df['xVelocity'] = dx
    df['yVelocity'] = dy
    df['xAcceleration'] = ddx
    df['yAcceleration'] = ddy
    return df

def resample_to_20hz(df):
    return df[df['Global_Time'] % 200 == 0].reset_index(drop=True)

def smooth_vehicle(df):
    df['Local_X'] = savgol_smooth(df['Local_X'].values)
    df['Local_Y'] = savgol_smooth(df['Local_Y'].values)
    df = compute_derivatives(df)
    return df

def process_file(i):
    num = f"{i:02d}_"
    df = pd.read_csv(os.path.join(RAW_PATH, num + "tracks.csv"))
    meta = pd.read_csv(os.path.join(RAW_PATH, num + "tracksMeta.csv"))
    rec = pd.read_csv(os.path.join(RAW_PATH, num + "recordingMeta.csv"))

    # Rename and convert
    df.rename(columns={
        'id': 'Vehicle_ID', 'frame': 'Frame_ID', 'x': 'Local_Y', 'y': 'Local_X',
        'width': 'v_Width', 'height': 'v_length', 'laneId': 'Lane_ID',
        'precedingId': 'Preceding', 'followingId': 'Following',
        'xVelocity': 'yVelocity', 'yVelocity': 'xVelocity',
        'xAcceleration': 'yAcceleration', 'yAcceleration': 'xAcceleration'
    }, inplace=True)

    # Convert to feet
    df['Local_X'] = df['Local_X'] / 0.3048
    df['Local_Y'] = df['Local_Y'] / 0.3048

    # Add missing fields
    df['Global_Time'] = (df['Frame_ID'] - 1) * 40 + i * 1000000000

    all_smoothed = []
    for vid in df['Vehicle_ID'].unique():
        vdf = df[df['Vehicle_ID'] == vid].copy()
        if len(vdf) < 5:
            continue
        smoothed = smooth_vehicle(vdf)
        all_smoothed.append(smoothed)

    if not all_smoothed:
        return

    final_df = pd.concat(all_smoothed, axis=0)
    final_df = resample_to_20hz(final_df)

    final_df = final_df[['Global_Time', 'Vehicle_ID', 'Frame_ID', 'Local_X', 'Local_Y',
                         'xVelocity', 'yVelocity', 'xAcceleration', 'yAcceleration',
                         'Lane_ID', 'Preceding', 'Following']]

    output_file = os.path.join(SMOOTHED_PATH, f"{num}smoothed_20hz.csv")
    final_df.to_csv(output_file, index=False)

def main():
    print("Smoothing and resampling highD dataset...")
    for i in tqdm(range(1, 61)):
        try:
            process_file(i)
        except Exception as e:
            print(f"Failed on {i:02d}: {e}")

if __name__ == "__main__":
    main()
