#!/usr/bin/env python3
"""Export NoMaD epoch summaries to publication figures without W&B.

Only the Python standard library and matplotlib are required. Accepts either
training.log (including tqdm output) or the existing metrics.jsonl file.
"""

import argparse
import ast
import csv
import json
import math
from pathlib import Path
import re
import sys


MODES = ("random masking", "no masking", "goal masking")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
SUMMARY = re.compile(r"Epoch\s+(\d+)\s+metrics:\s*(\{.*)")
FLOAT = r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf)"
BATCH_METRIC = re.compile(
    r"\(epoch\s+(?P<epoch>\d+)\)\s+\(batch\s+(?P<batch>\d+)/(?P<last_batch>\d+)\)\s+"
    r"(?P<metric>\w+)\s+\((?P<split>[^)]+)\):\s*(?P<value>" + FLOAT + r")"
    r"\s+\(\d+pt moving_avg:\s*(?P<moving_average>" + FLOAT + r")\)"
    r"\s+\(avg:\s*(?P<running_average>" + FLOAT + r")\)", re.IGNORECASE,
)
TRAJECTORY_METRICS = (
    "gc_dist_loss", "gc_action_loss", "uc_action_loss",
    "gc_action_waypts_cos_sim", "uc_action_waypts_cos_sim",
    "gc_multi_action_waypts_cos_sim", "uc_multi_action_waypts_cos_sim",
)


def read_metrics(path, batch_samples=None):
    """Read epoch summaries and sampled Logger lines, ignoring tqdm postfixes.

    Logger's last `avg` is its mean over ALL sampled batches in that epoch,
    not the moving average. Its printed precision is four decimal places.
    Exact epoch-summary metrics always take precedence over sampled metrics.
    """
    records = {}
    samples = {}
    duplicates = 0
    previous_epoch = -1
    with Path(path).open(encoding="utf-8", errors="replace") as stream:
        for line_number, line in enumerate(stream, 1):
            clean = ANSI.sub("", line).strip()
            sample = BATCH_METRIC.search(clean)
            if sample:
                row = sample.groupdict()
                for key in ("epoch", "batch", "last_batch"):
                    row[key] = int(row[key])
                for key in ("value", "moving_average", "running_average"):
                    row[key] = float(row[key])
                if row["batch"] > row["last_batch"]:
                    raise ValueError(f"Invalid batch index at line {line_number}.")
                if row["epoch"] < previous_epoch:
                    raise ValueError("Epoch numbers decreased: this file may contain multiple runs. "
                                     "Split it into separate runs before plotting.")
                previous_epoch = row["epoch"]
                samples[(row["epoch"], row["split"], row["metric"], row["batch"])] = row
            match = SUMMARY.search(clean)
            is_json = clean.startswith('{"epoch"') or (
                Path(path).suffix == ".jsonl" and clean.startswith("{")
            )
            if not match and not is_json:
                continue
            payload = match.group(2) if match else clean
            try:
                record = ast.literal_eval(payload) if match else json.loads(payload)
            except (ValueError, SyntaxError, json.JSONDecodeError) as exc:
                if not line.endswith("\n"):
                    print(f"Warning: skipped unfinished summary at line {line_number}.", file=sys.stderr)
                    continue
                raise ValueError(f"Invalid epoch summary at line {line_number}: {exc}") from exc
            epoch = record.get("epoch")
            if type(epoch) is not int or epoch < 0:
                raise ValueError(f"Invalid epoch at line {line_number}: {epoch!r}")
            if match and epoch != int(match.group(1)):
                raise ValueError(f"Mismatched epoch numbers at line {line_number}.")
            if epoch < previous_epoch:
                raise ValueError("Epoch numbers decreased: this file may contain multiple runs. "
                                 "Split it into separate runs before plotting.")
            previous_epoch = epoch
            duplicates += epoch in records
            records[epoch] = record
    if not records:
        raise ValueError("No completed 'Epoch N metrics: {...}' summaries found. "
                         "Use train/training.log or a NoMaD metrics.jsonl file.")
    if duplicates:
        print(f"Warning: kept the last summary for {duplicates} repeated epochs.", file=sys.stderr)
    # A summary marks the epoch as complete. Exclude any partially logged epoch.
    complete_samples = sorted((s for s in samples.values() if s["epoch"] in records),
                              key=lambda s: (s["epoch"], s["split"], s["metric"], s["batch"]))
    latest = {}
    for sample in complete_samples:
        latest[(sample["epoch"], sample["split"], sample["metric"])] = sample
    for (epoch, split, metric), sample in latest.items():
        records[epoch].setdefault(split, {}).setdefault(metric, sample["running_average"])
    if batch_samples is not None:
        batch_samples.extend(complete_samples)
    return [records[e] for e in sorted(records)]


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def flatten(record):
    row = {}
    for key, value in record.items():
        if isinstance(value, dict):
            for metric, result in value.items():
                row[f"{key}/{metric}"] = result
        else:
            row[key] = value
    return row


def eval_value(record, dataset, mode):
    return record.get(dataset, {}).get(f"diffusion_eval_loss ({mode})")


def macro(record, datasets, mode):
    values = [eval_value(record, dataset, mode) for dataset in datasets]
    if values and all(number(v) for v in values):
        return sum(values) / len(values)
    return None


def select_checkpoint(records, datasets):
    """Select a single epoch using the unweighted mean over all test datasets."""
    candidates = [(macro(r, datasets, MODES[0]), r) for r in records]
    candidates = [(value, r) for value, r in candidates if value is not None]
    return min(candidates, key=lambda pair: (pair[0], pair[1]["epoch"])) if candidates else (None, None)


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(text):
    replacements = {"\\": r"\textbackslash{}", "_": r"\_", "&": r"\&",
                    "%": r"\%", "#": r"\#", "$": r"\$", "{": r"\{", "}": r"\}"}
    return "".join(replacements.get(char, char) for char in text)


def export(records, source, output, formats, dpi, log_y, batch_samples=None, figures_only=False):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MaxNLocator
    except ImportError as exc:
        raise ValueError("matplotlib is required; run this script in the nomad_train environment.") from exc

    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 10,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.linewidth": 1.5, "savefig.dpi": dpi,
        "axes.prop_cycle": plt.cycler(color=["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]),
    })
    datasets = sorted({key for r in records for key, value in r.items()
                       if key != "train" and isinstance(value, dict)
                       and any(f"diffusion_eval_loss ({mode})" in value for mode in MODES)})
    files = []

    def draw(ax, getter, label, loss=True, **kwargs):
        # NaN preserves gaps at epochs where an evaluation was not recorded.
        points = [(r["epoch"], getter(r)) for r in records]
        positive_only = loss and log_y
        valid_count = sum(number(y) and (not positive_only or y > 0) for _, y in points)
        if valid_count:
            ax.plot([x for x, _ in points],
                    [y if number(y) and (not positive_only or y > 0) else math.nan for _, y in points],
                    label=label, marker="." if valid_count == 1 else None, **kwargs)

    def finish(fig, axes, name, ylabel, loss=True):
        for ax in axes:
            ax.set_xlabel("Epoch (zero-based)")
            ax.set_ylabel(ylabel)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(True, alpha=0.22, linewidth=0.5)
            if loss and log_y:
                ax.set_yscale("log")
            if ax.lines:
                ax.legend(frameon=False)
        if not any(ax.lines for ax in axes):
            plt.close(fig)
            return
        fig.tight_layout()
        for suffix in formats:
            path = output / f"{name}.{suffix}"
            fig.savefig(path, bbox_inches="tight")
            files.append(path.name)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.5))
    draw(ax, lambda r: r.get("train", {}).get("diffusion_loss"), "Train (raw weights)", color="#222222")
    for dataset in datasets:
        draw(ax, lambda r, d=dataset: eval_value(r, d, MODES[0]), dataset)
    finish(fig, [ax], "diffusion_loss", "Diffusion loss (random masking)")

    if datasets:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
        for mode, style in zip(MODES, ("-", "--", ":")):
            draw(ax, lambda r, m=mode: macro(r, datasets, m), mode, linestyle=style)
        finish(fig, [ax], "masking_comparison", "Macro-average diffusion loss (EMA)")

        columns = min(3, len(datasets))
        rows = math.ceil(len(datasets) / columns)
        fig, grid = plt.subplots(rows, columns, squeeze=False, figsize=(3.5 * columns, 2.8 * rows))
        axes = list(grid.flat)
        for ax, dataset in zip(axes, datasets):
            ax.set_title(dataset)
            for mode, style in zip(MODES, ("-", "--", ":")):
                draw(ax, lambda r, d=dataset, m=mode: eval_value(r, d, m), mode, linestyle=style)
        for ax in axes[len(datasets):]:
            ax.set_visible(False)
        finish(fig, axes[:len(datasets)], "per_dataset_masking", "Diffusion loss (EMA)")

    fig, grid = plt.subplots(1, 3, figsize=(10, 2.8))
    for ax, metric in zip(grid, ("total_loss", "diffusion_loss", "dist_loss")):
        draw(ax, lambda r, m=metric: r.get("train", {}).get(m), metric)
    finish(fig, list(grid), "training_losses", "Training loss")

    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    draw(ax, lambda r: r.get("lr"), "Learning rate used", loss=False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    finish(fig, [ax], "learning_rate", "Learning rate", loss=False)

    # Automatically export every recorded metric, with train/test split names.
    metric_splits = {}
    for record in records:
        for split, values in record.items():
            if isinstance(values, dict):
                for metric, value in values.items():
                    if number(value):
                        metric_splits.setdefault(metric, set()).add(split)
    for metric, splits in sorted(metric_splits.items()):
        fig, ax = plt.subplots(figsize=(5.6, 3.5))
        is_loss = "cos_sim" not in metric
        for split in sorted(splits, key=lambda s: (s != "train", s)):
            draw(ax, lambda r, s=split, m=metric: r.get(s, {}).get(m), split,
                 loss=is_loss, linestyle="--" if split == "train" else "-")
        if not is_loss:
            ax.set_ylim(-1.05, 1.05)
        slug = re.sub(r"[^a-zA-Z0-9_]+", "_", metric).strip("_")
        finish(fig, [ax], f"metric_{slug}", metric, loss=is_loss)

    if figures_only:
        print(f"Parsed {len(records)} completed epochs ({records[0]['epoch']}–{records[-1]['epoch']}).")
        print(f"Exported {len(files)} image files to {output.resolve()}")
        return

    samples = batch_samples or []
    if samples:
        write_csv(output / "batch_metrics.csv", samples)
    coverage = []
    for metric, splits in sorted(metric_splits.items()):
        for split in sorted(splits):
            epochs = [r["epoch"] for r in records if number(r.get(split, {}).get(metric))]
            sampled = [s for s in samples if s["split"] == split and s["metric"] == metric]
            coverage.append({"metric": metric, "split": split, "epochs": len(epochs),
                             "first_epoch": min(epochs), "last_epoch": max(epochs),
                             "logged_batch_samples": len(sampled)})
    if coverage:
        write_csv(output / "metric_coverage.csv", coverage)
    missing = [m for m in TRAJECTORY_METRICS if m not in metric_splits]
    if missing:
        print("Missing trajectory metrics (enable print_log_freq for the next run): "
              + ", ".join(missing), file=sys.stderr)

    flat = [flatten(r) for r in records]
    for row, record in zip(flat, records):
        for mode in MODES:
            row[f"macro/{mode}"] = macro(record, datasets, mode)
    write_csv(output / "epoch_metrics.csv", flat)

    best_value, best = select_checkpoint(records, datasets)
    if best is not None:
        table = [{"dataset": d, "epoch": best["epoch"],
                  **{m: eval_value(best, d, m) for m in MODES}} for d in datasets]
        table.append({"dataset": "Macro average", "epoch": best["epoch"],
                      **{m: macro(best, datasets, m) for m in MODES}})
        write_csv(output / "selected_checkpoint.csv", table)
        tex = ["% All rows use the SAME epoch, selected by macro random-masking loss.",
               r"\begin{tabular}{lrrr}", r"\hline",
               r"Dataset & Random masking & No masking & Goal masking \\", r"\hline"]
        for row in table:
            values = [f"{row[m]:.5f}" if number(row[m]) else "--" for m in MODES]
            tex.append(latex_escape(row["dataset"]) + " & " + " & ".join(values) + r" \\")
        tex += [r"\hline", r"\end{tabular}", f"% Selected zero-based epoch: {best['epoch']}"]
        (output / "selected_checkpoint.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")

    metadata = {
        "source": str(source.resolve()), "completed_epochs": len(records),
        "loss_axis_scale": "log" if log_y else "linear",
        "first_epoch": records[0]["epoch"], "last_epoch": records[-1]["epoch"],
        "datasets": datasets, "selected_epoch": best["epoch"] if best else None,
        "selected_macro_random_masking_loss": best_value, "figures": files,
        "missing_trajectory_metrics": missing,
        "logged_batch_samples": len(samples),
        "notes": [
            "Epoch summaries retain full precision; Logger batch lines have four-decimal precision.",
            "Sampled metrics use the final printed cumulative avg, never moving_avg or the final batch alone.",
            "Trajectory metrics average only sampled batches, not all batches; sample counts are in metric_coverage.csv.",
            "Sampled training trajectory metrics use evolving EMA weights during the epoch, not one final checkpoint.",
            "Cosine similarities use a linear [-1, 1] axis, including when --log-y is set.",
            "No smoothing or interpolation. Epochs are zero-based; incomplete epochs are omitted.",
            "Each logged epoch loss is the existing trainer's unweighted average of batch losses.",
            "Train uses raw weights; evaluation uses EMA weights and the configured eval_fraction.",
            "Macro means give equal weight to each dataset; all datasets must be present.",
            "The table uses one epoch selected by minimum macro random-masking evaluation loss.",
            "Selecting on test datasets makes them validation data; use a held-out test for final reporting.",
            "A single run provides no across-seed uncertainty; no confidence intervals are inferred.",
            "Losses do not measure navigation success rate or trajectory quality directly.",
        ],
    }
    (output / "run_summary.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Parsed {len(records)} completed epochs ({records[0]['epoch']}–{records[-1]['epoch']}).")
    print(f"Exported {len(files)} figures and metric tables to {output.resolve()}")
    if best:
        print(f"Selected epoch: {best['epoch']}; macro random-masking loss: {best_value:.6g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", type=Path, default=Path(__file__).with_name("training.log"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "logs" / "paper_figures")
    parser.add_argument("--formats", nargs="+", choices=("pdf", "png", "svg"), default=["pdf", "png"])
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--log-y", action="store_true", help="Use log loss axes; nonpositive losses are omitted.")
    parser.add_argument("--figures-only", action="store_true", help="Export only image files, without CSV/LaTeX/JSON tables.")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        samples = []
        records = read_metrics(args.log, samples)
        export(records, args.log, args.output_dir, args.formats, args.dpi, args.log_y, samples,
               figures_only=args.figures_only)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
