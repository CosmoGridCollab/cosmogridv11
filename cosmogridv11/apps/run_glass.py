"""Utility entry points for the GLASS pipeline.

This module mirrors the command line interface provided by
``run_probemaps`` but currently only implements the argument parsing and
basic application skeleton.
"""

from __future__ import annotations

import argparse
import os
from typing import Iterable, Sequence

from cosmogridv11 import utils_config, utils_io, utils_logging

LOGGER = utils_logging.get_logger(__file__)


def setup(args: Sequence[str]) -> argparse.Namespace:
    """Parse command line arguments for the GLASS pipeline.

    Parameters
    ----------
    args:
        Sequence of command line arguments excluding the program name.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with absolute paths resolved where applicable.
    """

    description = "Run GLASS pipeline tasks"
    parser = argparse.ArgumentParser(description=description, add_help=True)
    parser.add_argument(
        "-v",
        "--verbosity",
        type=str,
        default="info",
        choices=("critical", "error", "warning", "info", "debug"),
        help="logging level",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration yaml file",
    )
    parser.add_argument(
        "--dir_out",
        type=str,
        required=False,
        default=None,
        help=(
            "Output directory for the results; use None for the current "
            "directory."
        ),
    )
    parser.add_argument(
        "--num_maps_per_index",
        type=int,
        default=20,
        help="Number of permutations per index to process",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Enable test mode",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip if output file exists (useful for crashed jobs)",
    )
    parser.add_argument(
        "--dir_out_archive",
        type=str,
        default=None,
        help="Optional archive directory where results will be copied",
    )
    parser.add_argument(
        "--largemem",
        action="store_true",
        help="Request additional memory resources",
    )
    parser.add_argument(
        "--long",
        action="store_true",
        help="Request extended runtime",
    )
    parser.add_argument(
        "--precopy",
        action="store_true",
        help="Copy all simulations before processing",
    )

    parsed = parser.parse_args(args)

    utils_logging.set_all_loggers_level(parsed.verbosity)

    parsed.config = utils_io.get_abs_path(parsed.config)
    if parsed.dir_out is not None:
        parsed.dir_out = utils_io.get_abs_path(parsed.dir_out)
    if parsed.dir_out_archive is not None:
        parsed.dir_out_archive = utils_io.get_abs_path(parsed.dir_out_archive)

    return parsed


def main(indices: Iterable[int], args: Sequence[str]) -> None:
    """Entry point for GLASS processing jobs.

    Parameters
    ----------
    indices:
        Iterable of integer job indices to be processed.
    args:
        Command line arguments provided to the application.
    """

    parsed_args = setup(args)
    LOGGER.info("Starting GLASS pipeline stub")
    LOGGER.debug("Indices received: %s", list(indices))

    config = utils_config.load_config(parsed_args.config)
    LOGGER.debug("Configuration keys: %s", list(config))

    workdir = os.environ.get("TMPDIR", os.getcwd())
    if parsed_args.dir_out is None:
        parsed_args.dir_out = workdir

    LOGGER.info("Working directory: %s", workdir)
    LOGGER.info("Output directory: %s", parsed_args.dir_out)

    raise NotImplementedError(
        "GLASS processing is not yet implemented; this module currently "
        "provides only the command line interface stub."
    )
