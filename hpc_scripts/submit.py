import copy
import itertools
import os
import subprocess
from datetime import datetime

import numpy as np
import pandas as pd
import yaml


def get_default_config(experiment):
    with open(
        os.path.join("hpc_scripts", experiment, "default.yaml"), "r"
    ) as d_conf:
        return yaml.safe_load(d_conf)


def create_config(experiment, default_config, **kwargs):
    config = copy.deepcopy(default_config)
    config.update(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, (list, tuple, set)):
            kwargs[k] = "_".join([str(w) for w in v])
    parameter_str = "+".join([f"{k}@{v}" for k, v in kwargs.items()])
    new_config_path = os.path.join(
        "hpc_scripts", experiment, f"{parameter_str}.yaml"
    )
    new_bs_config_path = os.path.join(
        "hpc_scripts", experiment, f"{parameter_str}__bs.yaml"
    )

    with open(new_config_path, "w") as new_conf:
        yaml.dump(config, new_conf)
    return new_config_path, new_bs_config_path, parameter_str


def submit_pbs_job(
    experiment,
    train_file,
    val_file,
    param_combination,
    default_config,
    echo_only,
):
    new_config_path, new_bs_config_path, parameter_str = create_config(
        experiment, default_config, **param_combination
    )

    output_dir = os.path.join("logs", experiment, parameter_str)
    pbs_log_file = os.path.join(
        output_dir, f"pbs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.out"
    )

    env_vars = {
        "TRAIN_FILE": train_file,
        "VAL_FILE": val_file,
        "OUTPUT_DIR": output_dir,
        "CONFIG_FILE": new_config_path,
        "BS_CONFIG_FILE": new_bs_config_path,
    }

    pbs_file = os.path.join("hpc_scripts", experiment, f"{experiment}.pbs")

    command = [
        "qsub",
        "-o",
        pbs_log_file,
        "-j",
        "oe",
        pbs_file,
        "-v",
        ",".join(f"{k}={v}" for k, v in env_vars.items()),
    ]

    subprocess.run(["echo"] + command, check=True)
    if not echo_only:
        subprocess.run(command, check=True)


def submit_slurm_job(
    experiment,
    train_file,
    val_file,
    param_combination,
    default_config,
    echo_only,
):
    new_config_path, new_bs_config_path, parameter_str = create_config(
        experiment, default_config, **param_combination
    )

    output_dir = os.path.join("logs", experiment, parameter_str)
    slurm_log_file = os.path.join(
        output_dir, f"slurm_{datetime.now().strftime('%Y%m%d-%H%M%S')}.out"
    )

    env_vars = {
        "TRAIN_FILE": train_file,
        "VAL_FILE": val_file,
        "OUTPUT_DIR": output_dir,
        "CONFIG_FILE": new_config_path,
        "BS_CONFIG_FILE": new_bs_config_path,
    }

    slurm_file = os.path.join(
        "hpc_scripts", experiment, f"{experiment}.sbatch"
    )

    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in env_vars.items())

    command = [
        "sbatch",
        "--output",
        slurm_log_file,
        "--export",
        export_str,
        slurm_file,
    ]

    subprocess.run(["echo"] + command, check=True)
    if not echo_only:
        subprocess.run(command, check=True)


def submit_grid_commands(
    experiment,
    train_file,
    val_file,
    dynamic_params=None,
    exclude=None,
    echo_only=False,
    use_slurm=True,
    **kwargs,
):
    default_config = get_default_config(experiment)
    kwargs = dict(sorted(kwargs.items()))

    submit_func = submit_slurm_job if use_slurm else submit_pbs_job

    for param_combination in itertools.product(*kwargs.values()):
        param_combination = dict(zip(kwargs.keys(), param_combination))

        if exclude is not None and param_combination in exclude:
            continue

        if dynamic_params is not None:
            for p, c in dynamic_params.items():
                param_combination[p] = c(param_combination)

        submit_func(
            experiment,
            train_file,
            val_file,
            param_combination,
            default_config,
            echo_only,
        )


def submit_hpt_commands(
    experiment, train_file, val_file, hpt_ids, echo_only=False
):
    default_config = get_default_config(experiment)
    hpt_file = os.path.join("hpt", experiment, f"configurations.csv")
    hpt_df = pd.read_csv(hpt_file, index_col=0)

    for hpt_id in hpt_ids:
        param_combination = hpt_df.loc[hpt_id]
        param_combination = param_combination.to_dict()
        param_combination.pop("log_dir")
        param_combination.pop("valid_CELoss")

        submit_job(
            experiment,
            train_file,
            val_file,
            param_combination,
            default_config,
            echo_only,
        )


if __name__ == "__main__":
    # train_file = "massivekb_data/scaling_data_max_100000/train_2s_1000000p.mgf"
    # val_file = "massivekb_data/scaling_data_max_100000/val_0.25.mgf"

    # submit_grid_commands(
    #     experiment="old_optim_scheduler",
    #     train_file=train_file,
    #     val_file=val_file,
    #     learning_rate=[1e-4, 1.6e-4, 2.5e-4, 4e-4, 6.3e-4, 1e-3],
    # )
    # submit_hpt_commands(
    #     experiment="lr_scheduler",
    #     train_file=train_file,
    #     val_file=val_file,
    #     hpt_ids=range(21, 26),
    # )
    # submit_hpt_commands(
    #     experiment="bs_lr_default",
    #     train_file=train_file,
    #     val_file=val_file,
    #     hpt_ids=range(20, 25),
    # )
    # submit_grid_commands(
    #     experiment="introducing_new",
    #     train_file=train_file,
    #     val_file=val_file,
    #     lr_scheduler=["onecycle"],
    #     learning_rate=[float(2 ** (-11.5)), float(2 ** (-11.25))],
    #     cycle_momentum=[False],
    # )
    # submit_grid_commands(
    #     experiment="introducing_new",
    #     train_file=train_file,
    #     val_file=val_file,
    #     lr_scheduler=["onecycle", "cosinewarmup"],
    #     learning_rate=[float(2 ** (-11.5)), float(2 ** (-11.25))],
    # )

    # submit_grid_commands(
    #     experiment="introducing_new",
    #     train_file=train_file,
    #     val_file=val_file,
    #     lr_scheduler=["onecycle"],
    #     learning_rate=[
    #         float(2**p) for p in np.arange(-11.25, -10.25 + 0.25, 0.25)
    #     ],
    #     gradient_clip_val=[0.5, 1, 2],
    #     gradient_clip_algorithm=["norm"],
    # )

    # submit_grid_commands(
    #     experiment="optimizer_2",
    #     train_file=train_file,
    #     val_file=val_file,
    #     optimizer=["Adam", "AdamW"],
    #     betas=[[0.9, b2] for b2 in [0.98, 0.99, 0.999]],
    #     weight_decay=[2e-6, 1e-5, 5e-5],
    #     optimizer_eps=[1e-8],
    #     learning_rate=[
    #         float(2**p) for p in np.arange(-11.25, -10.25 + 0.25, 0.25)
    #     ],
    # )

    # submit_grid_commands(
    #     experiment="label_smoothing",
    #     train_file=train_file,
    #     val_file=val_file,
    #     train_label_smoothing=[0.0, 0.001, 0.01, 0.03, 0.05, 0.1],
    #     learning_rate=[
    #         float(2**p) for p in np.arange(-11.25, -10.25 + 0.25, 0.25)
    #     ],
    # )

    ####
    # MassIVE-KB v2: Model Scaling

    train_file = (
        "massivekb_data/massiveKB_3cac0386/subsets/train_20s_4243254p.mgf"
    )
    val_file = "massivekb_data/massiveKB_3cac0386/subsets/val.mgf"

    submit_grid_commands(
        experiment="bs_lr_S",
        train_file=train_file,
        val_file=val_file,
        dynamic_params={
            "max_steps": lambda c: 192_000_000 // c["global_train_batch_size"],
            "val_check_interval": lambda c: c["max_steps"] // 20,
        },
        global_train_batch_size=[2**x for x in range(11, 12)],
        learning_rate=[float(10**p) for p in np.arange(-5, -2.5 + 0.5, 0.5)],
    )
