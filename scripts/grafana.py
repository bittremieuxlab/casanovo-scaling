import datetime
import hashlib
import json
import os
import subprocess

import pandas as pd
import yaml
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

STATE_FILE = "logs/metrics_state.json"


def setup_db(
    token="2mBaDlegqP4bIeal1s6EXm5ciwWh-hfeboKHwulUIm7peWti49_PCiab-K7hkdYBmzyenOw0b0F0RckzvskDDg==",
    org="casanovo-scaling",
):
    url = "http://localhost:8086"

    client = InfluxDBClient(url=url, token=token, org=org)
    return client


def file_hash(path):
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


def load_metrics(csv_path, keys, keep_nans=False):
    df = pd.read_csv(csv_path)
    dfs = {}
    for key in keys:
        if key in df.columns:
            if keep_nans:
                dfs[key] = df
            else:
                dfs[key] = df[df[key].notna()]
    return dfs


def sync_metrics(dbclient, log_dir, experiment_name, metric_keys):
    write_api = dbclient.write_api(write_options=SYNCHRONOUS)
    state = load_state()
    updated_state = {}

    for run in os.listdir(log_dir):
        csv_path = os.path.join(log_dir, run, "csv_logs", "metrics.csv")
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}")
            continue

        current_hash = file_hash(csv_path)
        if state.get(os.path.join(log_dir, run)) == current_hash:
            # Skip unchanged files
            print(f"{os.path.join(log_dir, run)} unchanged, skipping")
            continue

        print(f"Processing {os.path.join(log_dir, run)}")
        metrics_dfs = load_metrics(
            csv_path,
            metric_keys,
        )
        points = []
        for metric_type, metric_df in metrics_dfs.items():
            for _, row in metric_df.iterrows():
                start_time = datetime.datetime(2025, 8, 1)
                point_time = start_time + datetime.timedelta(
                    seconds=int(row.step)
                )
                points.append(
                    Point(experiment_name)
                    .tag("run", run)
                    .tag("type", metric_type)
                    .field("step", row.step)
                    .field("value", row[metric_type])
                    .time(point_time)
                )
        if points:
            write_api.write(
                bucket="casanovo-scaling",
                org="casanovo-scaling",
                record=points,
            )

        updated_state[os.path.join(log_dir, run)] = current_hash

    save_state({**state, **updated_state})


def sync_logs(
    remote="vsc20683@tier1.hpc.ugent.be:/dodrio/scratch/projects/2025_048/casanovo-scaling/logs/",
    local="/home/pigeonmark/Documents/casanovo-scaling/logs/",
    exclude="*.ckpt",
):
    rsync_command = [
        "rsync",
        "-avz",
        "--update",
        f"--exclude={exclude}",
        remote,
        local,
    ]
    print("Running rsync to sync logs. This may take a while...")
    subprocess.run(rsync_command, check=True)
    print("Sync complete.")


def verify_runs(experiment_name):
    log_dir = os.path.join("logs", experiment_name)
    for run in os.listdir(log_dir):
        csv_path = os.path.join(log_dir, run, "csv_logs", "metrics.csv")
        config_path = os.path.join(
            "hpc_scripts", experiment_name, f"{run}.yaml"
        )
        if not os.path.exists(csv_path):
            print(f"Could not find {csv_path}")
            continue

        if not os.path.exists(config_path):
            print(f"Could not find {config_path}")
            continue

        metrics_dfs = load_metrics(
            csv_path,
            ["valid_CELoss"],
            keep_nans=True,
        )
        config = yaml.safe_load(open(config_path, "r"))

        expected_steps = config["max_steps"]
        max_valid_steps = max(metrics_dfs["valid_CELoss"]["step"])

        if max_valid_steps + 1 < expected_steps:
            print(
                f"Run {run} only did {max_valid_steps + 1} instead of {expected_steps} steps."
            )


if __name__ == "__main__":
    sync_logs()
    sync_logs(
        remote="vsc20683@tier1.hpc.ugent.be:/dodrio/scratch/projects/2025_048/casanovo-scaling/hpc_scripts/",
        local="/home/pigeonmark/Documents/casanovo-scaling/hpc_scripts/",
    )
    dbclient = setup_db()
    # sync_metrics(
    #     dbclient,
    #     experiment_name="train_subsets",
    #     log_dir="logs/casanovo_train_subsets/",
    #     metric_keys=["lr-Adam", "train_CELoss", "valid_CELoss"],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="lr_scheduler",
    #     log_dir="logs/lr_scheduler/",
    #     metric_keys=[
    #         "lr-AdamW",
    #         "lr-AdamW-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="steps",
    #     log_dir="logs/steps/",
    #     metric_keys=[
    #         "lr-AdamW",
    #         "lr-AdamW-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="old_optim_scheduler",
    #     log_dir="logs/old_optim_scheduler/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-Adam-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="bs_lr_default",
    #     log_dir="logs/bs_lr_default/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-Adam-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )

    # sync_metrics(
    #     dbclient,
    #     experiment_name="introducing_new",
    #     log_dir="logs/introducing_new/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-Adam-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="optimizer",
    #     log_dir="logs/optimizer/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-AdamW",
    #         "lr-Adam-momentum",
    #         "lr-AdamW-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    # sync_metrics(
    #     dbclient,
    #     experiment_name="optimizer_2",
    #     log_dir="logs/optimizer_2/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-AdamW",
    #         "lr-Adam-momentum",
    #         "lr-AdamW-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )
    #
    # sync_metrics(
    #     dbclient,
    #     experiment_name="label_smoothing",
    #     log_dir="logs/label_smoothing/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-Adam-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )

    # sync_metrics(
    #     dbclient,
    #     experiment_name="v2_train_subsets",
    #     log_dir="logs/v2_train_subsets/",
    #     metric_keys=[
    #         "lr-Adam",
    #         "lr-Adam-momentum",
    #         "train_CELoss_step",
    #         "valid_CELoss",
    #     ],
    # )

    sync_metrics(
        dbclient,
        experiment_name="bs_lr_S",
        log_dir="logs/bs_lr_S/",
        metric_keys=[
            "lr-Adam",
            "lr-Adam-momentum",
            "train_CELoss_step",
            "valid_CELoss",
        ],
    )

    print()
    print()
    verify_runs(experiment_name="bs_lr_S")
