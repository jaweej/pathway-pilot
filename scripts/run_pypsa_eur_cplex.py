"""Set up and run a pinned PyPSA-Eur electricity instance with CPLEX."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess


PYPSA_EUR_VERSION = "v2026.02.0"
PYPSA_EUR_COMMIT = "d6383ebf602767b1adbb676fe8a16e37a6e9f932"
WORKFLOW_DIRS = ("benchmarks", "cutouts", "logs", "resources", "results")


@dataclass(frozen=True)
class Instance:
    """Files and resources required by one supported PyPSA-Eur instance."""

    config_name: str
    run_name: str
    network_name: str
    mem_mb: int
    cores: int

    @property
    def target(self) -> str:
        return f"results/{self.run_name}/networks/{self.network_name}"

    @property
    def prepare_target(self) -> str:
        return f"resources/{self.run_name}/networks/{self.network_name}"


INSTANCES = {
    "tutorial": Instance(
        config_name="config.cplex-tutorial.yaml",
        run_name="cplex-tutorial",
        network_name="base_s_5_elec_.nc",
        mem_mb=12000,
        cores=1,
    ),
    "elec-50-12h": Instance(
        config_name="config.cplex-elec-50-12h.yaml",
        run_name="cplex-elec-50-12h",
        network_name="base_s_50_elec_12h.nc",
        mem_mb=24000,
        cores=2,
    ),
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _dev_data_dir() -> Path:
    configured = os.getenv("PATHWAY_PILOT_DEV_DATA_DIR")
    return Path(configured) if configured else Path("C:/Users/B510067/dev_data")


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _ensure_checkout(project_root: Path) -> Path:
    checkout = project_root / "pypsa-eur"
    if not checkout.exists():
        _run(
            [
                "git",
                "clone",
                "--branch",
                PYPSA_EUR_VERSION,
                "--depth",
                "1",
                "https://github.com/PyPSA/pypsa-eur.git",
                str(checkout),
            ],
            cwd=project_root,
        )
    if not (checkout / ".git").exists():
        raise RuntimeError(f"{checkout} exists but is not a PyPSA-Eur git checkout")

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != PYPSA_EUR_COMMIT:
        raise RuntimeError(
            f"Expected PyPSA-Eur {PYPSA_EUR_COMMIT}, but {checkout} is at {revision}"
        )
    return checkout


def _remove_empty_directory(path: Path) -> None:
    if not path.is_dir() or path.is_symlink():
        return
    entries = list(path.iterdir())
    if all(entry.name == ".gitkeep" and entry.is_file() for entry in entries):
        for entry in entries:
            entry.unlink()
        path.rmdir()
    elif entries:
        raise RuntimeError(
            f"Refusing to replace populated directory {path} with a DEV_DATA_DIR junction"
        )
    else:
        path.rmdir()


def _ensure_junction(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    if link.exists():
        try:
            if link.resolve() == target.resolve():
                return
        except OSError:
            pass
        _remove_empty_directory(link)
    if link.exists():
        raise RuntimeError(f"Refusing to replace existing path {link}")
    _run(
        [
            "powershell.exe",
            "-NoProfile",
            "-Command",
            "New-Item -ItemType Junction -Path $args[0] -Target $args[1] | Out-Null",
            str(link),
            str(target),
        ],
        cwd=link.parent,
    )


def _dataset_names(checkout: Path) -> list[str]:
    versions = checkout / "data" / "versions.csv"
    with versions.open(encoding="utf-8", newline="") as handle:
        return sorted({row["dataset"] for row in csv.DictReader(handle)})


def _set_up_storage(checkout: Path, data_root: Path) -> None:
    for name in WORKFLOW_DIRS:
        _ensure_junction(checkout / name, data_root / name)
    for dataset in _dataset_names(checkout):
        _ensure_junction(checkout / "data" / dataset, data_root / "data" / dataset)
    for name in (".snakemake-runtime-cache", "localappdata", "solver-work", "tmp"):
        (data_root / name).mkdir(parents=True, exist_ok=True)


def _install_integration(
    project_root: Path,
    checkout: Path,
    data_root: Path,
    instance: Instance,
) -> Path:
    integration = project_root / "integrations" / "pypsa-eur" / PYPSA_EUR_VERSION
    patch = integration / "solve-network-cplex.patch"
    reverse_check = subprocess.run(
        ["git", "apply", "--reverse", "--check", str(patch)],
        cwd=checkout,
        capture_output=True,
    )
    if reverse_check.returncode != 0:
        _run(["git", "apply", "--check", str(patch)], cwd=checkout)
        _run(["git", "apply", str(patch)], cwd=checkout)

    template = (integration / instance.config_name).read_text(encoding="utf-8")
    rendered = template.replace(
        "__PYPSA_EUR_DATA_ROOT__", data_root.resolve().as_posix()
    )
    config = checkout / "config" / instance.config_name
    config.write_text(rendered, encoding="utf-8")
    return config


def _run_workflow(
    project_root: Path,
    checkout: Path,
    data_root: Path,
    config: Path,
    instance: Instance,
    *,
    prepare_only: bool,
) -> None:
    snakemake = project_root / ".venv" / "Scripts" / "snakemake.exe"
    if not snakemake.is_file():
        raise RuntimeError(
            "Snakemake is missing. Install requirements-pypsa-eur.txt in the local .venv."
        )
    environment = os.environ.copy()
    environment["TEMP"] = str(data_root / "tmp")
    environment["TMP"] = str(data_root / "tmp")
    environment["LOCALAPPDATA"] = str(data_root / "localappdata")
    target = instance.prepare_target if prepare_only else instance.target
    command = [
        str(snakemake),
        "-c",
        "2" if prepare_only else str(instance.cores),
        target,
        "--configfile",
        str(config),
        "--runtime-source-cache-path",
        str(data_root / ".snakemake-runtime-cache"),
        "--show-failed-logs",
        "--rerun-incomplete",
        "--resources",
        f"mem_mb={instance.mem_mb}",
    ]
    subprocess.run(command, cwd=checkout, env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance",
        choices=sorted(INSTANCES),
        default="tutorial",
        help="PyPSA-Eur instance to set up and run (default: tutorial).",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Create the checkout, integration, and DEV_DATA_DIR junctions without running.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Build the prepared PyPSA network but do not optimize it.",
    )
    args = parser.parse_args()
    instance = INSTANCES[args.instance]

    project_root = _project_root()
    data_root = _dev_data_dir() / "pathway-pilot" / f"pypsa-eur-{PYPSA_EUR_VERSION}"
    checkout = _ensure_checkout(project_root)
    _set_up_storage(checkout, data_root)
    config = _install_integration(project_root, checkout, data_root, instance)
    if not args.setup_only:
        _run_workflow(
            project_root,
            checkout,
            data_root,
            config,
            instance,
            prepare_only=args.prepare_only,
        )
    print(f"PyPSA-Eur checkout: {checkout}")
    print(f"PyPSA-Eur data root: {data_root}")
    print(f"Result network: {data_root / Path(instance.target)}")


if __name__ == "__main__":
    main()
