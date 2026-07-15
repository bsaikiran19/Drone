#!/usr/bin/env python3
"""
plot_results.py

Reads a single mission_log_*.csv (produced by mission_logger.MissionLogger)
and generates IEEE-quality, 300 DPI PNG figures.

Only figures backed by real collected data are generated:
  1. Distance to Goal vs Time
  2. UAV Linear Velocity vs Time
  3. UAV Angular Velocity vs Time
  4. Detection Count vs Time
  5. Object Class Distribution
  6. Navigation Decision Distribution
  7. Tracking Count vs Time
  8. YOLO Inference Time vs Time
  9. YOLO FPS vs Time
  10. Mission Success Summary

If a metric column is entirely blank/NA in the CSV (e.g. no objects were
ever detected during the run, so Object_Classes is empty for every row),
the corresponding figure is skipped with a printed explanation rather than
being drawn empty or filled with placeholder data.

Usage:
    python3 plot_results.py --csv logs/mission_log_20260715_143000.csv
    python3 plot_results.py --log-dir logs   # auto-picks the most recent file
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

# ── IEEE-style rcParams ──────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 0.8,
    'lines.linewidth': 1.2,
})

IEEE_1COL = (3.45, 2.6)  # inches, single-column IEEE figure size


def find_latest_csv(log_dir):
    files = sorted(glob.glob(os.path.join(log_dir, 'mission_log_*.csv')))
    if not files:
        raise FileNotFoundError(f'No mission_log_*.csv files found in {log_dir}')
    return files[-1]


def savefig(fig, out_dir, name):
    path = os.path.join(out_dir, f'{name}.png')
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {path}')


def has_data(series):
    return series is not None and series.notna().any() and len(series.dropna()) > 0


# ── 1. Distance to Goal vs Time ──────────────────────────────────────────

def plot_distance_to_goal(df, out_dir):
    if not has_data(df.get('Distance_To_Goal')):
        print('  skip fig_distance_to_goal: Distance_To_Goal has no data')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(df['Mission_Time'], df['Distance_To_Goal'], color='tab:red')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Distance to goal (m)')
    ax.set_title('Distance-to-Goal Convergence')
    savefig(fig, out_dir, 'fig_distance_to_goal')


# ── 2. UAV Linear Velocity vs Time ───────────────────────────────────────

def plot_linear_velocity(df, out_dir):
    cols = ['Linear_Velocity_X', 'Linear_Velocity_Y', 'Linear_Velocity_Z']
    if not any(has_data(df.get(c)) for c in cols):
        print('  skip fig_linear_velocity: no linear velocity data')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(df['Mission_Time'], df['Linear_Velocity_X'], label='X', color='tab:blue')
    ax.plot(df['Mission_Time'], df['Linear_Velocity_Y'], label='Y', color='tab:orange')
    ax.plot(df['Mission_Time'], df['Linear_Velocity_Z'], label='Z', color='tab:green')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Commanded linear velocity (m/s)')
    ax.set_title('UAV Linear Velocity vs. Time')
    ax.legend(loc='best', ncol=3)
    savefig(fig, out_dir, 'fig_linear_velocity')


# ── 3. UAV Angular Velocity vs Time ──────────────────────────────────────

def plot_angular_velocity(df, out_dir):
    if not has_data(df.get('Angular_Velocity_Z')):
        print('  skip fig_angular_velocity: no angular velocity data')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(df['Mission_Time'], df['Angular_Velocity_Z'], color='tab:red')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Commanded angular velocity (rad/s)')
    ax.set_title('UAV Angular Velocity vs. Time')
    savefig(fig, out_dir, 'fig_angular_velocity')


# ── 4. Detection Count vs Time ───────────────────────────────────────────

def plot_detection_count(df, out_dir):
    if not has_data(df.get('Detection_Count')):
        print('  skip fig_detection_count: no detection count data')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(df['Mission_Time'], df['Detection_Count'], color='tab:blue')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Detected objects')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title('Detection Count vs. Time')
    savefig(fig, out_dir, 'fig_detection_count')


# ── 5. Object Class Distribution ─────────────────────────────────────────

def plot_object_class_distribution(df, out_dir):
    classes_col = df.get('Object_Classes')
    if classes_col is None or classes_col.fillna('').eq('').all():
        print('  skip fig_object_class_distribution: no objects were ever '
              'detected in this run (Object_Classes is empty for all rows)')
        return

    counts = {}
    for cell in classes_col.dropna():
        if not cell:
            continue
        for cls in str(cell).split(';'):
            counts[cls] = counts.get(cls, 0) + 1

    if not counts:
        print('  skip fig_object_class_distribution: no non-empty class entries')
        return

    fig, ax = plt.subplots(figsize=IEEE_1COL)
    names = list(counts.keys())
    values = [counts[n] for n in names]
    ax.bar(names, values, color='tab:blue')
    ax.set_ylabel('Rows containing this class')
    ax.set_title('Detected Object Class Distribution')
    ax.tick_params(axis='x', rotation=45)
    savefig(fig, out_dir, 'fig_object_class_distribution')


# ── 6. Navigation Decision Distribution ──────────────────────────────────

def plot_navigation_decision_distribution(df, out_dir):
    if not has_data(df.get('Navigation_Decision')):
        print('  skip fig_navigation_decision_distribution: no data')
        return
    counts = df['Navigation_Decision'].value_counts()
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.bar(counts.index, counts.values, color='tab:purple')
    ax.set_ylabel('Row count')
    ax.set_title('Navigation Decision Distribution')
    ax.tick_params(axis='x', rotation=30)
    savefig(fig, out_dir, 'fig_navigation_decision_distribution')


# ── 7. Tracking Count vs Time ────────────────────────────────────────────

def plot_tracking_count(df, out_dir):
    if not has_data(df.get('Tracked_Object_Count')):
        print('  skip fig_tracking_count: no tracked-object-count data')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(df['Mission_Time'], df['Tracked_Object_Count'], color='tab:orange')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Active Kalman tracks')
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.set_title('Tracking Count vs. Time')
    savefig(fig, out_dir, 'fig_tracking_count')


# ── 8. YOLO Inference Time vs Time ───────────────────────────────────────

def plot_yolo_inference_time(df, out_dir):
    d = df[df['YOLO_Inference_Time_ms'].notna()] if 'YOLO_Inference_Time_ms' in df else None
    if d is None or d.empty:
        print('  skip fig_yolo_inference_time: no inference-time samples '
              '(YOLO never ran, or model failed to load)')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(d['Mission_Time'], d['YOLO_Inference_Time_ms'], color='tab:brown')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('Inference time (ms)')
    ax.set_title('YOLOv8 Inference Time vs. Time')
    savefig(fig, out_dir, 'fig_yolo_inference_time')


# ── 9. YOLO FPS vs Time ──────────────────────────────────────────────────

def plot_yolo_fps(df, out_dir):
    d = df[df['YOLO_FPS'].notna()] if 'YOLO_FPS' in df else None
    if d is None or d.empty:
        print('  skip fig_yolo_fps: no FPS samples (YOLO never ran, or '
              'model failed to load)')
        return
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    ax.plot(d['Mission_Time'], d['YOLO_FPS'], color='tab:green')
    ax.set_xlabel('Mission time (s)')
    ax.set_ylabel('FPS')
    ax.set_title('YOLOv8 Inference FPS vs. Time')
    savefig(fig, out_dir, 'fig_yolo_fps')


# ── 10. Mission Success Summary ──────────────────────────────────────────

def plot_mission_success_summary(df, out_dir):
    if not has_data(df.get('Mission_Status')):
        print('  skip fig_mission_success_summary: no Mission_Status data')
        return
    counts = df['Mission_Status'].value_counts()
    fig, ax = plt.subplots(figsize=IEEE_1COL)
    colors = {'RUNNING': 'tab:blue', 'SUCCESS': 'tab:green'}
    ax.bar(counts.index, counts.values,
           color=[colors.get(k, 'gray') for k in counts.index])
    ax.set_ylabel('Row count (ticks)')
    ax.set_title('Mission Status Summary (this run)')
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom', fontsize=8)
    savefig(fig, out_dir, 'fig_mission_success_summary')

    final_status = df['Mission_Status'].iloc[-1]
    print(f'  final Mission_Status of this run: {final_status}')
    print('  NOTE: this run-level bar chart shows tick counts, not a '
          'multi-trial success RATE. This codebase has no failure/crash '
          'detector, so only RUNNING/SUCCESS states can ever appear here — '
          'see metrics_report.md for that caveat if you generated it '
          'separately. A cross-trial success-rate figure requires '
          'aggregating the final row of multiple real mission_log CSVs.')


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default=None, help='Path to a specific mission_log_*.csv')
    parser.add_argument('--log-dir', default='logs', help='Directory to search for the latest mission_log_*.csv')
    parser.add_argument('--out-dir', default='./figures')
    args = parser.parse_args()

    csv_path = args.csv or find_latest_csv(args.log_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    print(f'Loaded {csv_path} ({len(df)} rows)\n')

    plot_distance_to_goal(df, args.out_dir)
    plot_linear_velocity(df, args.out_dir)
    plot_angular_velocity(df, args.out_dir)
    plot_detection_count(df, args.out_dir)
    plot_object_class_distribution(df, args.out_dir)
    plot_navigation_decision_distribution(df, args.out_dir)
    plot_tracking_count(df, args.out_dir)
    plot_yolo_inference_time(df, args.out_dir)
    plot_yolo_fps(df, args.out_dir)
    plot_mission_success_summary(df, args.out_dir)

    print(f'\nDone. Figures written to: {os.path.abspath(args.out_dir)}')


if __name__ == '__main__':
    main()
