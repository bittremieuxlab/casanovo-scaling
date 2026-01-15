import os
import random
import re
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
import pandas as pd
from datasail.sail import datasail
from pyteomics import mgf
from tqdm import tqdm

from massivekb_data.dist_matrix import rectangle_distance_matrix


def remove_ptms(seq):
    return "".join([c for c in seq if "A" <= c <= "Z"])


def get_train_val_test(split_df):
    train = split_df[split_df["split"] == "train"]
    val = split_df[split_df["split"] == "val"]
    test = split_df[split_df["split"] == "test"]
    return train, val, test


def create_sub_mgf(mgf_file, sub_cache_dir, num_spectra=1000):
    sub_spectra = []
    with mgf.read(
        mgf_file,
        use_index=False,
        convert_arrays=0,
        read_charges=False,
        read_ions=False,
    ) as massivekb:
        for i, spectrum in enumerate(massivekb):
            if i == num_spectra:
                break
            sub_spectra.append(spectrum)
        sub_file = os.path.join(sub_cache_dir, f"sub_{num_spectra}.mgf")
        mgf.write(sub_spectra, sub_file)
    return sub_file


def create_sequence_index(mgf_file, cache_dir=None, total=None):
    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, "sequence_index.csv")
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

    with mgf.read(
        mgf_file,
        use_index=False,
        convert_arrays=0,
        read_charges=False,
        read_ions=False,
    ) as massivekb:
        data = []
        failed = 0
        for i, spectrum in enumerate(tqdm(massivekb, total=total)):
            if spectrum is None:
                continue
            try:
                params = spectrum["params"]
                seq = params["seq"]

                data.append(
                    {
                        "mgf_i": i,
                        "sequence": seq,
                        "unmodified_sequence": remove_ptms(seq),
                        "title": params["title"],
                        "charge": int(params["charge"][0]),
                        "mass": params["pepmass"][0],
                        "rt": params["rtinseconds"],
                    }
                )
            except KeyError as e:
                print(spectrum["params"])
                failed += 1
                continue
    print(f"Got {failed} failed spectra in total")
    df = pd.DataFrame(data)

    if cache_dir is not None:
        cache_file = os.path.join(cache_dir, "sequence_index.csv")
        df.to_csv(cache_file, index=False)

    return df


def split_datasail(
    sequences,
    dist_matrix,
    splits,
    e_clusters=200,
    epsilon=0.1,
    threads=cpu_count(),
    cache_dir=None,
    from_cache=True,
):
    n = len(sequences)
    cache_file = os.path.join(cache_dir, f"split_df_{n}.csv")

    if cache_dir is not None and from_cache:
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

    e_splits, _, _ = datasail(
        verbose="D",
        techniques=["C1e"],
        splits=splits,
        names=["train", "val", "test"],
        runs=1,
        e_type="O",
        epsilon=epsilon,
        e_clusters=e_clusters,
        e_data={i: s for i, s in enumerate(sequences)},
        e_dist=(list(range(n)), dist_matrix),
        threads=threads,
    )

    df = pd.DataFrame.from_dict(
        e_splits["C1e"][0], orient="index", columns=["split"]
    )
    df["sequence"] = sequences

    if cache_dir is not None:
        df.to_csv(cache_file, index=False)
    return df


def add_all_to_split(
    split_df,
    unique_sequences,
    n_threads=cpu_count(),
    cache_dir=None,
    from_cache=True,
):
    cache_file = os.path.join(cache_dir, f"full_split_df_{len(split_df)}.csv")
    if cache_dir is not None and from_cache:
        if os.path.exists(cache_file):
            return pd.read_csv(cache_file)

    dist_matrix = rectangle_distance_matrix(
        unique_sequences, split_df["sequence"].values, n_threads
    )

    nearest_indices = np.argmin(dist_matrix, axis=1)
    split_labels = split_df["split"].values[nearest_indices]

    full_split_df = pd.DataFrame(
        {"sequence": unique_sequences, "split": split_labels}
    )

    if cache_dir is not None:
        full_split_df.to_csv(cache_file, index=False)
    return full_split_df


def create_val_test_traini(
    mgf_file, full_split_df, exclude_ptms, output_dir, total=None
):
    ptm_pattern = re.compile("|".join(map(re.escape, exclude_ptms)))
    match = ptm_pattern.search

    def contains_exclude_ptm(sequence: str) -> bool:
        return bool(match(sequence))

    split_dict = dict(zip(full_split_df["sequence"], full_split_df["split"]))

    spectra = {k: [] for k in ["val", "test", "rare_PTM"]}
    train_index = defaultdict(list)
    print("Indexing mgf file, this might take a while")
    indexed_mgf_reader = mgf.read(
        mgf_file,
        use_index=True,
        convert_arrays=0,
        read_charges=False,
        read_ions=False,
    )
    for spectrum in tqdm(indexed_mgf_reader, total=total):
        seq = spectrum["params"]["seq"]
        if contains_exclude_ptm(seq):
            spectra["rare_PTM"].append(spectrum)
            continue

        unmod_pep = remove_ptms(seq)
        split = split_dict[unmod_pep]

        if split == "train":
            train_index[seq].append(spectrum["params"]["title"])
        else:
            spectra[split].append(spectrum)

    os.makedirs(output_dir, exist_ok=True)
    mgf.write(spectra["val"], os.path.join(output_dir, f"val.mgf"))
    mgf.write(spectra["test"], os.path.join(output_dir, f"test.mgf"))
    mgf.write(spectra["rare_PTM"], os.path.join(output_dir, f"rare_ptm.mgf"))

    return indexed_mgf_reader, train_index


def create_train_subsets(
    indexed_mgf_reader, train_index, n_train_spectra, n_train_peps, output_dir
):
    spectra_count_dict = {p: len(l) for p, l in train_index.items()}

    for n_train_s in n_train_spectra:
        # Get all peptides with enough spectra
        p_with_enough_spectra = [
            p for p, c in spectra_count_dict.items() if c >= n_train_s
        ]
        remaining_peptides = [
            p for p, c in spectra_count_dict.items() if c < n_train_s
        ]

        print(
            f"There are {len(p_with_enough_spectra)} peptides with at least {n_train_s} spectra"
        )

        for n_train_p in n_train_peps:
            if n_train_p is None:
                n_train_p = len(spectra_count_dict.keys())
            print(f"Getting {n_train_s} spectra for {n_train_p} peptides")
            if len(spectra_count_dict.keys()) < n_train_p:
                train_spectra = []
                print(
                    f"There are only {len(spectra_count_dict.keys())} peptides but requested {n_train_p}"
                )

            elif len(p_with_enough_spectra) >= n_train_p:
                # Sample random peptides and spectra
                train_peptides = random.sample(
                    p_with_enough_spectra, n_train_p
                )
                train_spectra = [
                    spec_i
                    for train_pep in train_peptides
                    for spec_i in random.sample(
                        train_index[train_pep], n_train_s
                    )
                ]
                print(
                    f"Sufficient peptides with sufficient spectra, number of spectra selected: {len(train_spectra)}"
                )
            else:
                # Get all peptides with enough spectra and sample random spectra
                train_spectra = [
                    spec_i
                    for train_pep in p_with_enough_spectra
                    for spec_i in random.sample(
                        train_index[train_pep], n_train_s
                    )
                ]
                print(
                    f"Got {len(train_spectra)} spectra from peptides with enough spectra"
                )

                # Count how many are missing
                n_train_p_to_add = n_train_p - len(p_with_enough_spectra)

                # Select random peptides to add
                train_p_to_add = random.sample(
                    remaining_peptides, n_train_p_to_add
                )

                # Add all spectra for these random peptides
                train_spectra += [
                    spec_i
                    for train_pep in train_p_to_add
                    for spec_i in train_index[train_pep]
                ]
                print(
                    f"Added all spectra from random peptides, selected {len(train_spectra)} spectra in total now"
                )

            # Shuffle, then get spectra from indices and write to mgf
            random.shuffle(train_spectra)

            def train_spectra_generator():
                for title in tqdm(
                    train_spectra,
                    desc=f"Writing train_{n_train_s}s_{n_train_p}p.mgf",
                ):
                    yield indexed_mgf_reader.get_spectrum(title)

            mgf.write(
                train_spectra_generator(),
                os.path.join(
                    output_dir, f"train_{n_train_s}s_{n_train_p}p.mgf"
                ),
            )
            print()
