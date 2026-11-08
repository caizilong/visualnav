"""Fit R1 spatial bins and a constant visitation prior from train trajectories."""

import argparse
import json
from pathlib import Path

import yaml

from .simulation import load_manifest
from .trajectory_regions import fit_region_statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Habitat training YAML, with distance/action bounds")
    parser.add_argument("--manifest", help="Override simulation.manifest")
    parser.add_argument("--output", help="Override simulation.r1_statistics")
    args = parser.parse_args()
    defaults = Path(__file__).resolve().parents[2] / "config/defaults.yaml"
    config = yaml.safe_load(defaults.read_text())
    config.update(yaml.safe_load(Path(args.config).read_text()))
    if config.get("training_profile") != "habitat_scratch":
        parser.error("R1 statistics require the habitat_scratch training configuration")
    manifest = args.manifest or config["simulation"]["manifest"]
    destination = args.output or config["simulation"].get("r1_statistics")
    if not destination:
        parser.error("Provide --output or simulation.r1_statistics")
    bundle = load_manifest(manifest, config["context_size"])
    statistics = fit_region_statistics(bundle, config)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(statistics, indent=2, allow_nan=False) + "\n")
    print(f"Wrote R1 schema 2 bins and train-only prior to {output}; {statistics['reservoir_points']} sampled waypoints")


if __name__ == "__main__":
    main()
