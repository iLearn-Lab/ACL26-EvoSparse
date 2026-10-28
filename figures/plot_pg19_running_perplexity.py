"""Plot running PG-19 perplexity from per-token negative log likelihoods."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DENSE = REPO_ROOT / "results" / "pg19_dense_cod_speed" / "log.txt"
DEFAULT_EVOSPARSE = (
    REPO_ROOT
    / "results"
    / "pg19_evosparse_cod_speed_b4096_bs16_h20_g15"
    / "log.txt"
)


def load_running_perplexity(path: Path, max_tokens: int) -> np.ndarray:
    """Load NLL values and return exp(cumulative mean NLL)."""
    values = np.loadtxt(path, dtype=np.float64, max_rows=max_tokens, ndmin=1)
    if values.size == 0:
        raise ValueError(f"No NLL values found in {path}")
    cumulative_mean = np.cumsum(values) / np.arange(1, values.size + 1)
    return np.exp(cumulative_mean)


def thousands(value: float, _position: int) -> str:
    if value == 0:
        return "0"
    return f"{value / 1000:g}k"


def build_figure(
    dense_path: Path,
    evosparse_path: Path,
    output_stem: Path,
    max_tokens: int = 32_000,
    display_start: int = 1_000,
    budget: int = 4_096,
) -> None:
    dense = load_running_perplexity(dense_path, max_tokens)
    evosparse = load_running_perplexity(evosparse_path, max_tokens)
    count = min(dense.size, evosparse.size, max_tokens)
    if display_start >= count:
        raise ValueError("display_start must be smaller than the number of tokens")

    dense = dense[:count]
    evosparse = evosparse[:count]
    token_ids = np.arange(1, count + 1)
    visible = token_ids >= display_start

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(6.8, 4.2), constrained_layout=True)
    dense_color = "#0072B2"
    evosparse_color = "#D55E00"

    ax.axvspan(display_start, budget, color="#F2F2F2", zorder=0)
    ax.axvline(budget, color="#777777", linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
    ax.text(
        budget + 350,
        0.98,
        "KV budget = 4,096",
        color="#555555",
        fontsize=9,
        va="top",
        transform=ax.get_xaxis_transform(),
    )

    ax.plot(
        token_ids[visible],
        dense[visible],
        color=dense_color,
        linewidth=2.0,
        label="Full attention",
        zorder=3,
    )
    ax.plot(
        token_ids[visible],
        evosparse[visible],
        color=evosparse_color,
        linewidth=2.0,
        linestyle=(0, (5, 2)),
        label="EvoSparse",
        zorder=4,
    )

    ax.scatter([count], [dense[-1]], color=dense_color, s=24, zorder=5)
    ax.scatter([count], [evosparse[-1]], color=evosparse_color, s=24, zorder=5)

    combined = np.concatenate([dense[visible], evosparse[visible]])
    padding = max(0.025, (combined.max() - combined.min()) * 0.12)
    ax.set_ylim(combined.min() - padding, combined.max() + padding)
    ax.set_xlim(display_start, count + 1_600)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_major_formatter(FuncFormatter(thousands))

    ax.annotate(
        f"Full: {dense[-1]:.3f}",
        xy=(count, dense[-1]),
        xytext=(10, -13),
        textcoords="offset points",
        color=dense_color,
        va="center",
        fontsize=9,
    )
    ax.annotate(
        f"Evo: {evosparse[-1]:.3f}",
        xy=(count, evosparse[-1]),
        xytext=(10, 13),
        textcoords="offset points",
        color=evosparse_color,
        va="center",
        fontsize=9,
    )

    ax.set_title("Running perplexity on PG-19")
    ax.set_xlabel("Evaluated context length (tokens)")
    ax.set_ylabel("Perplexity (lower is better)")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper right", frameon=False)

    delta = evosparse[-1] - dense[-1]
    relative = 100.0 * delta / dense[-1]
    ax.text(
        0.02,
        0.04,
        f"32k tokens: ΔPPL = {delta:+.4f} ({relative:+.3f}%)",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dense", type=Path, default=DEFAULT_DENSE)
    parser.add_argument("--evosparse", type=Path, default=DEFAULT_EVOSPARSE)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "figures" / "pg19_running_perplexity",
        help="Output path without a suffix.",
    )
    parser.add_argument("--max_tokens", type=int, default=32_000)
    parser.add_argument("--display_start", type=int, default=1_000)
    parser.add_argument("--budget", type=int, default=4_096)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_figure(
        args.dense,
        args.evosparse,
        args.output,
        max_tokens=args.max_tokens,
        display_start=args.display_start,
        budget=args.budget,
    )
