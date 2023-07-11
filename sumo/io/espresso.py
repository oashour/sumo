"""
IO module for Quantum ESPRESSO
"""
import os
import math
from copy import deepcopy
import errno
import logging
import sys

from pymatgen.io.espresso.inputs import PWin
from pymatgen.io.espresso.inputs.pwin import KPointsCard, AdditionalKPointsCard


def write_kpoint_files(
    filename,
    kpoints,
    labels,
    make_folders=False,
    hybrid=False,
    kpts_per_split=None,
    directory=None,
    cart_coords=False,
):
    r"""Write the k-points data to PW input files.

    Files are named {input_filename}_band_split_01.{input_filename_extension}


    Args:
        filename (:obj:`str`): Path to PWscf input file
        kpoints (:obj:`numpy.ndarray`): The k-point coordinates along the
            high-symmetry path. For example::

                [[0, 0, 0], [0.25, 0, 0], [0.5, 0, 0], [0.5, 0, 0.25],
                [0.5, 0, 0.5]]

        labels (:obj:`list`) The high symmetry labels for each k-point (will be
            an empty :obj:`str` if the k-point has no label). For example::

                ['\Gamma', '', 'X', '', 'Y']

        make_folders (:obj:`bool`, optional): Generate folders and copy in
            required files (INCAR, POTCAR, POSCAR, and possibly CHGCAR) from
            the current directory.

        hybrid (:obj:`bool`, optional): Whether the calculation is a hybrid
            functional calculation. Defaults to ``False``.
            If `True`, the k-points are given weight 0 and added to the
            `ADDITIONAL_K_POINTS` card.

        kpts_per_split (:obj:`int`, optional): If set, the k-points are split
            into separate input files each containing the number
            of k-points specified. This is useful for hybrid band structure
            calculations (or large supercells) where it is often intractable
            to calculate all k-points in the same calculation.

        directory (:obj:`str`, optional): The output file directory.

        cart_coords (:obj:`bool`, optional): Whether the k-points are returned
            in cartesian or reciprocal coordinates. Defaults to ``False``
            (fractional coordinates). # TODO: not yet implemented
    """
    kpt_splits, label_splits, weight_splits, option = split_kpoints(
        kpoints, labels, kpts_per_split, cart_coords, hybrid
    )

    pwin_files = prepare_pwins(
        filename, kpt_splits, label_splits, weight_splits, option, hybrid
    )

    write_pwin_files(filename, directory, pwin_files, make_folders)


def split_kpoints(kpoints, labels, kpts_per_split, cart_coords, hybrid):
    """
    Split k-points into separate input files
    Args:
        kpoints (:obj:`numpy.ndarray`): The k-point coordinates along the
            high-symmetry path.
        labels (:obj:`list`) The high symmetry labels for each k-point
        kpts_per_split (:obj:`int`): Number of k-points per split
        cart_coords (:obj:`bool`): Whether the k-points are returned in
            cartesian or reciprocal coordinates.
    Returns:
        :obj:`tuple`: Tuple containing:
            :obj:`list`: List of k-point splits
            :obj:`list`: List of label splits
            :obj:`list`: List of weight splits
            :obj:`KPointsOptions`: K-point option
    """
    Card = AdditionalKPointsCard if hybrid else KPointsCard
    if kpts_per_split:
        kpt_splits = [
            kpoints[i : i + kpts_per_split]
            for i in range(0, len(kpoints), kpts_per_split)
        ]
        label_splits = [
            labels[i : i + kpts_per_split]
            for i in range(0, len(labels), kpts_per_split)
        ]
        weight_splits = [
            [0 if hybrid else 1] * len(split) for split in kpt_splits
        ]
        option = Card.opts.tpiba if cart_coords else Card.opts.crystal
    else:
        hsp_indices = [i for i, label in enumerate(labels) if label != ""]
        kpt_splits = [[kpoints[i] for i in hsp_indices]]
        label_splits = [[labels[i] for i in hsp_indices]]
        weight_splits = [
            [
                hsp_indices[i + 1] - hsp_indices[i]
                for i in range(len(hsp_indices) - 1)
            ]
            + [1]
        ]  # Last point's weight in *_b modes is always 1
        option = Card.opts.tpiba_b if cart_coords else Card.opts.crystal_b

    return kpt_splits, label_splits, weight_splits, option


def prepare_pwins(
    filename, kpt_splits, label_splits, weight_splits, option, hybrid
):
    """
    Prepare PWin objects for writing to file
    Args:
        filename (:obj:`str`): Path to original PWscf input file
        kpt_splits (:obj:`list`): List of k-point splits
        label_splits (:obj:`list`): List of label splits
        weight_splits (:obj:`list`): List of weight splits
        option (:obj:`KPointsOptions`): K-point option
    Returns:
        :obj:`list`: List of PWin objects
    """
    pwin_files = []
    parent_pwin = PWin.from_file(filename)
    for kpt_split, label_split, weight_split in zip(
        kpt_splits, label_splits, weight_splits
    ):
        pwin = deepcopy(parent_pwin)
        if hybrid:
            pwin.additional_k_points = AdditionalKPointsCard(
                option,
                k=kpt_split,
                weights=weight_split,
                labels=label_split,
            )
        if not hybrid:
            pwin.k_points = KPointsCard(
                option,
                grid=[],
                shift=[],
                k=kpt_split,
                weights=weight_split,
                labels=label_split,
            )
            if pwin.system is None:
                pwin.system = pwin.namelist_classes.system.value()
            pwin.control['calculation'] = "bands"
        pwin_files.append(pwin)
    return pwin_files


def write_pwin_files(filename, directory, pwin_files, make_folders):
    """
    Write the k-points data to PW input files.
    Args:
        filename (:obj:`str`): Path to original PWscf input file
        directory (:obj:`str`): The output file directory.
        pwin_files (:obj:`list`): List of PWin objects
    """

    pad = int(math.floor(math.log10(len(pwin_files)))) + 2
    for i, pwin_file in enumerate(pwin_files):
        if make_folders:
            folder = f"split-{str(i + 1).zfill(pad)}"
            folder = os.path.join(directory, folder) if directory else folder
            pwi_filename = os.path.join(folder, os.path.basename(filename))
            try:
                os.makedirs(folder, exist_ok=False)
            except OSError as e:  # pylint: disable=invalid-name
                if e.errno == errno.EEXIST:
                    logging.error(
                        "ERROR: Folders already exist, won't overwrite."
                    )
                else:
                    raise
                sys.exit()
        else:
            basename, extension = os.path.splitext(os.path.basename(filename))
            pwi_filename = (
                f"{basename}_band_split_{i + 1:0d}{extension}"
                if len(pwin_files) > 1
                else f"{basename}_band{extension}"
            )
            if directory:
                pwi_filename = os.path.join(directory, pwi_filename)
        pwin_file.to_file(pwi_filename)
