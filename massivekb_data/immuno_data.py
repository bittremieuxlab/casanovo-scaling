import os
from collections import Counter

import ppx


def get_full_extension(filename):
    """
    Extract the full extension, handling double extensions like .mzid.gz.
    Example:
      file.mzid.gz -> .mzid.gz
      file.mzML -> .mzml
      file.raw -> .raw
    """
    filename = filename.lower()
    base, ext1 = os.path.splitext(filename)
    base, ext2 = os.path.splitext(base)
    if ext1 and ext2:
        return ext2 + ext1  # e.g. .mzid + .gz -> .mzid.gz
    else:
        return ext1 or ext2  # single extension case


def process_remote_files(files):
    file_list = []

    for f in files:
        ext = get_full_extension(f)
        file_list.append((f, ext))

    ext_counts = Counter(ext for _, ext in file_list)

    # --- Print formatted summary ---
    print(f"Total files: {len(files)}")

    for ext, count in sorted(ext_counts.items()):
        print(f"  {ext}: {count}")
    print("\n-------------------------\n")


def get_pxd(pxd):
    try:
        proj = ppx.find_project(pxd)
        print(f"Dataset: {pxd}")
        print([ins.get('name') for ins in proj.metadata.get('instruments', [])])
    except ValueError as e:
        print(e)
        return
    files = proj.remote_files()
    process_remote_files(files)


def collect_psms(pxd_list):
    for pxd in pxd_list:
        get_pxd(pxd)


if __name__ == "__main__":
    immuno_pxds = [
        # Cell lines:
        "PXD045796",
        "PXD034059",
        "PXD006939",
        "PXD040858",
        "PXD031451",
        "PXD058303",
        "PXD040740",
        # Tissue samples:
        "PXD028309",
        "PXD038782",
        "PXD020186",
        "PXD019643",
        "PXD029567",
        "PXD026184",
        "PXD009738",
        "PXD015249",
        "PXD047311"
    ]
    collect_psms(immuno_pxds)
