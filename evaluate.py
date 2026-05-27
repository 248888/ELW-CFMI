"""Train Pattern-wise ELW-CFMI on the Section 4.1 simulation and evaluate it.

Metrics follow the paper's simulation study:

* standardized squared energy distance against an independent complete sample
* 0.1 quantile estimate of X1

By default, the script uses the known simulation propensity scores and passes
them to ``pattern_wise_elw_cfmi.py``. Use ``--propensity-mode estimated`` to
estimate propensities with ``pattern_wise_elw.py``, or ``--propensity-mode none``
to train the unweighted CFMI baseline.
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import json
import time
from typing import Any

import numpy as np

from pattern_wise_elw_cfmi import CFMIConfig
from pattern_wise_elw_cfmi import train_pattern_wise_elw_cfmi
from sample import SimulationDataset
from sample import load_simulation_dataset
from sample import make_simulation_dataset
from sample import missing_rate
from sample import true_propensity_by_observed_pattern


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def standardize_from_reference(
    x: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    """Standardize columns using means/stds from the complete reference data."""

    x = np.asarray(x, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    mean = reference.mean(axis=0)
    std = reference.std(axis=0)
    std = np.where(std > 1e-12, std, 1.0)
    return (x - mean[None, :]) / std[None, :]


def _maybe_subsample(
    x: np.ndarray,
    max_n: int | None,
    rng: np.random.Generator,
) -> np.ndarray:
    if max_n is None or x.shape[0] <= max_n:
        return x
    index = rng.choice(x.shape[0], size=max_n, replace=False)
    return x[index]


def mean_pairwise_l2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """Compute mean Euclidean distance over all cross pairs."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("x and y must have shapes (n, d) and (m, d).")

    total = 0.0
    count = 0
    for start in range(0, x.shape[0], chunk_size):
        block = x[start : start + chunk_size]
        distances = np.linalg.norm(block[:, None, :] - y[None, :, :], axis=2)
        total += float(distances.sum())
        count += block.shape[0] * y.shape[0]
    return total / count


def energy_distance_squared(
    x: np.ndarray,
    y: np.ndarray,
    *,
    chunk_size: int = 512,
) -> float:
    """V-statistic estimate of e^2(X, Y)."""

    cross = mean_pairwise_l2(x, y, chunk_size=chunk_size)
    xx = mean_pairwise_l2(x, x, chunk_size=chunk_size)
    yy = mean_pairwise_l2(y, y, chunk_size=chunk_size)
    return float(max(2.0 * cross - xx - yy, 0.0))


def evaluate_imputations(
    completed: np.ndarray,
    x_test: np.ndarray,
    reference: np.ndarray,
    *,
    quantile: float = 0.1,
    true_quantile: float = 0.1,
    energy_max_n: int | None = None,
    energy_chunk_size: int = 512,
    seed: int = 0,
) -> dict[str, float]:
    """Evaluate one or multiple completed datasets."""

    completed = np.asarray(completed)
    if completed.ndim == 2:
        completed = completed[None, :, :]
    if completed.ndim != 3:
        raise ValueError("completed must have shape (n, d) or (l, n, d).")

    rng = np.random.default_rng(seed)
    standardized_test = standardize_from_reference(x_test, reference)
    standardized_test = _maybe_subsample(standardized_test, energy_max_n, rng)

    energies = []
    quantiles = []
    for draw in completed:
        standardized_completed = standardize_from_reference(draw, reference)
        standardized_completed = _maybe_subsample(standardized_completed, energy_max_n, rng)
        energies.append(
            energy_distance_squared(
                standardized_completed,
                standardized_test,
                chunk_size=energy_chunk_size,
            )
        )
        quantiles.append(float(np.quantile(draw[:, 0], quantile)))

    energies_np = np.asarray(energies, dtype=float)
    quantiles_np = np.asarray(quantiles, dtype=float)
    errors = quantiles_np - true_quantile

    return {
        "energy_distance": float(energies_np.mean()),
        "energy_distance_std": float(energies_np.std(ddof=0)),
        "quantile": float(quantiles_np.mean()),
        "quantile_std": float(quantiles_np.std(ddof=0)),
        "quantile_error": float(errors.mean()),
        "quantile_abs_error": float(np.abs(errors).mean()),
    }


def run_replication(
    dataset: SimulationDataset,
    *,
    seed: int,
    hidden_dim: int,
    hidden_layers: int,
    steps: int,
    batch_size: int,
    lr: float,
    ode_steps: int,
    num_imputations: int,
    target_probability: float,
    propensity_mode: str,
    cross_fit: int,
    energy_max_n: int | None,
    energy_chunk_size: int,
    quiet: bool,
    device: str | None,
) -> dict[str, Any]:
    config = CFMIConfig(
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        steps=steps,
        batch_size=batch_size,
        lr=lr,
        target_probability=target_probability,
        seed=seed,
        device=device,
    )

    fit_kwargs: dict[str, Any] = {}
    if propensity_mode == "true":
        fit_kwargs["propensity_by_pattern"] = true_propensity_by_observed_pattern(
            dataset.pattern_probabilities
        )
    elif propensity_mode == "estimated":
        fit_kwargs["estimate_propensity"] = True
        fit_kwargs["propensity_kwargs"] = {
            "cross_fit": cross_fit,
            "random_state": seed,
        }
    elif propensity_mode == "none":
        pass
    else:
        raise ValueError(f"Unknown propensity_mode: {propensity_mode}")

    start = time.perf_counter()
    model, history = train_pattern_wise_elw_cfmi(
        dataset.x_missing,
        mask=dataset.observed_mask,
        config=config,
        verbose=not quiet,
        log_interval=max(1, steps // 10),
        **fit_kwargs,
    )
    train_seconds = time.perf_counter() - start

    start = time.perf_counter()
    completed = model.impute(
        dataset.x_missing,
        mask=dataset.observed_mask,
        num_imputations=num_imputations,
        ode_steps=ode_steps,
        seed=seed + 3_000_003,
    )
    impute_seconds = time.perf_counter() - start

    metrics = evaluate_imputations(
        completed,
        dataset.x_test,
        dataset.x_full,
        quantile=0.1,
        true_quantile=dataset.true_quantile_x1,
        energy_max_n=energy_max_n,
        energy_chunk_size=energy_chunk_size,
        seed=seed + 4_000_003,
    )

    return {
        **metrics,
        "n": int(dataset.x_full.shape[0]),
        "test_n": int(dataset.x_test.shape[0]),
        "distribution": dataset.distribution,
        "corr": float(dataset.corr),
        "train_seconds": float(train_seconds),
        "impute_seconds": float(impute_seconds),
        "final_loss": float(history[-1]["loss"]) if history else None,
        "history": history,
        "missing_rate": missing_rate(dataset.observed_mask),
        "n_patterns": int(len(model.elw_result_.pattern_results)) if model.elw_result_ else 0,
    }


def summarize_replications(replications: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = [
        "energy_distance",
        "quantile",
        "quantile_error",
        "quantile_abs_error",
        "train_seconds",
        "impute_seconds",
        "final_loss",
        "missing_rate",
    ]
    summary: dict[str, float] = {}
    for name in metric_names:
        values = np.asarray(
            [rep[name] for rep in replications if rep.get(name) is not None],
            dtype=float,
        )
        if values.size == 0:
            continue
        summary[f"{name}_mean"] = float(values.mean())
        summary[f"{name}_std"] = float(values.std(ddof=0))
    return summary


def main() -> None:
    parser = ArgumentParser(description="Evaluate ELW-CFMI on the Section 4.1 simulation.")
    parser.add_argument("--data", type=Path, default=None, help="Optional .npz from sample.py.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--test-n", type=int, default=None)
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--distribution", choices=["uniform", "gaussian"], default="uniform")
    parser.add_argument("--corr", type=float, default=0.7)
    parser.add_argument("--propensity-mode", choices=["true", "estimated", "none"], default="true")
    parser.add_argument("--cross-fit", type=int, default=5)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--target-probability", type=float, default=0.5)
    parser.add_argument("--ode-steps", type=int, default=50)
    parser.add_argument("--num-imputations", type=int, default=1)
    parser.add_argument("--energy-max-n", type=int, default=None)
    parser.add_argument("--energy-chunk-size", type=int, default=512)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    if args.data is not None and args.reps != 1:
        raise ValueError("--data can currently be used only with --reps 1.")

    replications: list[dict[str, Any]] = []
    for rep in range(args.reps):
        rep_seed = args.seed + rep
        if args.data is None:
            dataset = make_simulation_dataset(
                n=args.n,
                test_n=args.test_n,
                corr=args.corr,
                seed=rep_seed,
                distribution=args.distribution,
            )
        else:
            dataset = load_simulation_dataset(args.data)

        result = run_replication(
            dataset,
            seed=rep_seed,
            hidden_dim=args.hidden_dim,
            hidden_layers=args.hidden_layers,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            ode_steps=args.ode_steps,
            num_imputations=args.num_imputations,
            target_probability=args.target_probability,
            propensity_mode=args.propensity_mode,
            cross_fit=args.cross_fit,
            energy_max_n=args.energy_max_n,
            energy_chunk_size=args.energy_chunk_size,
            quiet=args.quiet,
            device=args.device,
        )
        result["rep"] = rep
        replications.append(result)
        if not args.quiet:
            print(
                "rep={rep} energy={energy:.6f} q10={q:.6f} abs_qerr={qe:.6f}".format(
                    rep=rep,
                    energy=result["energy_distance"],
                    q=result["quantile"],
                    qe=result["quantile_abs_error"],
                )
            )

    output = {
        "settings": {
            "n": replications[0]["n"],
            "test_n": replications[0]["test_n"],
            "reps": args.reps,
            "seed": args.seed,
            "distribution": replications[0]["distribution"],
            "corr": replications[0]["corr"],
            "propensity_mode": args.propensity_mode,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "hidden_layers": args.hidden_layers,
            "lr": args.lr,
            "ode_steps": args.ode_steps,
            "num_imputations": args.num_imputations,
        },
        "replications": replications,
        "summary": summarize_replications(replications),
    }

    text = json.dumps(output, indent=2, default=_json_default)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
