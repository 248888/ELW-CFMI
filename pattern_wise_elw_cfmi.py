"""Pattern-wise ELW-CFMI with a simple MLP velocity field.

The implementation follows the pseudo-code in
``pattern_wise_elw_cfmi_reformatted.pdf``:

1. Split samples by observed-missing pattern.
2. Estimate sample-level empirical likelihood weights inside each pattern.
3. Train one shared conditional flow-matching velocity field.
4. Impute missing values by Euler integration from Gaussian noise.

Input convention
----------------
``x`` is an ``(n, d)`` NumPy array. Missing entries can be encoded as ``NaN``.
Alternatively pass a boolean ``mask`` with ``True`` for observed entries.

The MLP input is the concatenation of the zero-filled target path state,
the zero-filled conditioning vector, the condition mask, the target mask,
and time ``tau``. Including the target mask is a practical disambiguation for
zero-valued target states while preserving the shared CFMI velocity field.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Literal
from typing import Mapping
from typing import Sequence
import warnings

import numpy as np
import torch
from torch import nn


PatternKey = tuple[int, ...]
TargetMoments = np.ndarray | Mapping[PatternKey, np.ndarray]
BalanceFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, PatternKey],
    tuple[np.ndarray, np.ndarray],
]


@dataclass
class PatternWeightResult:
    """Container for pattern-wise sample weights.

    Attributes
    ----------
    sample_weights:
        Global training weights ``a_i = rho_m * omega_i^(m)``. They sum to 1.
    within_pattern_weights:
        Pattern-local ELW weights. Each vector sums to 1.
    pattern_weights:
        Pattern weights ``rho_m``.
    pattern_indices:
        Row indices belonging to each pattern.
    """

    sample_weights: np.ndarray
    within_pattern_weights: dict[PatternKey, np.ndarray]
    pattern_weights: dict[PatternKey, float]
    pattern_indices: dict[PatternKey, np.ndarray]


@dataclass
class CFMIConfig:
    hidden_dim: int = 128
    hidden_layers: int = 3
    activation: Literal["relu", "silu", "gelu", "tanh"] = "silu"
    dropout: float = 0.0
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 128
    steps: int = 2_000
    target_probability: float = 0.5
    ensure_condition_when_possible: bool = True
    grad_clip_norm: float | None = 1.0
    standardize: bool = True
    device: str | torch.device | None = None
    seed: int | None = None


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unknown activation: {name}")


class SimpleMLPVelocity(nn.Module):
    """Simple MLP velocity field ``v_theta`` for CFMI."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        hidden_layers: int = 3,
        activation: str = "silu",
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        model_input_dim = 4 * input_dim + 1

        layers: list[nn.Module] = []
        prev_dim = model_input_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_target_path: torch.Tensor,
        x_condition: torch.Tensor,
        condition_mask: torch.Tensor,
        target_mask: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        if tau.ndim == 1:
            tau = tau[:, None]
        features = torch.cat(
            [
                x_target_path,
                x_condition,
                condition_mask.float(),
                target_mask.float(),
                tau.float(),
            ],
            dim=-1,
        )
        return self.net(features)


def group_by_pattern(mask: np.ndarray) -> dict[PatternKey, np.ndarray]:
    """Group row indices by observed-missing pattern."""

    mask_bool = np.asarray(mask, dtype=bool)
    groups: dict[PatternKey, list[int]] = {}
    for i, row in enumerate(mask_bool):
        key = tuple(int(v) for v in row)
        groups.setdefault(key, []).append(i)
    return {key: np.asarray(indices, dtype=np.int64) for key, indices in groups.items()}


def empirical_likelihood_weights(
    features: np.ndarray,
    target_moment: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-9,
    ridge: float = 1e-8,
    min_denom: float = 1e-8,
) -> np.ndarray:
    """Estimate EL weights under ``sum_i w_i features_i = target_moment``.

    The empirical likelihood solution has
    ``w_i = 1 / (n * (1 + lambda^T h_i))`` with
    ``h_i = features_i - target_moment``. A damped Newton iteration solves for
    ``lambda``. If the moment constraint is infeasible or numerically unstable,
    the caller should catch the warning/fallback behavior upstream.
    """

    g = np.asarray(features, dtype=np.float64)
    mu = np.asarray(target_moment, dtype=np.float64).reshape(-1)
    if g.ndim != 2:
        raise ValueError("features must have shape (n_samples, n_features).")
    if g.shape[1] != mu.shape[0]:
        raise ValueError("target_moment length must match features.shape[1].")
    if not np.isfinite(g).all() or not np.isfinite(mu).all():
        raise ValueError("features and target_moment must be finite.")

    n, p = g.shape
    if p == 0:
        return np.full(n, 1.0 / n, dtype=np.float64)

    h = g - mu[None, :]
    scale = h.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    h = h / scale[None, :]

    if np.linalg.norm(h.mean(axis=0)) < tol:
        return np.full(n, 1.0 / n, dtype=np.float64)

    lam = np.zeros(p, dtype=np.float64)
    last_score_norm = np.inf
    converged = False

    for _ in range(max_iter):
        denom = 1.0 + h @ lam
        if np.any(denom <= min_denom):
            raise RuntimeError("ELW Newton iterate left the feasible region.")

        score = (h / denom[:, None]).mean(axis=0)
        score_norm = float(np.linalg.norm(score))
        if score_norm < tol:
            converged = True
            break

        jac = -np.einsum("ni,nj,n->ij", h, h, 1.0 / (denom**2)) / n
        jac = jac - ridge * np.eye(p)
        try:
            step = np.linalg.solve(jac, -score)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError("ELW Newton system is singular.") from exc

        accepted = False
        step_scale = 1.0
        for _ in range(30):
            candidate = lam + step_scale * step
            candidate_denom = 1.0 + h @ candidate
            if np.any(candidate_denom <= min_denom):
                step_scale *= 0.5
                continue
            candidate_score = (h / candidate_denom[:, None]).mean(axis=0)
            candidate_norm = float(np.linalg.norm(candidate_score))
            if candidate_norm < min(last_score_norm, score_norm):
                lam = candidate
                last_score_norm = candidate_norm
                accepted = True
                break
            step_scale *= 0.5

        if not accepted:
            raise RuntimeError("ELW Newton line search failed.")

    if not converged:
        denom = 1.0 + h @ lam
        if np.any(denom <= min_denom):
            raise RuntimeError("ELW did not converge to a feasible solution.")
        score = (h / denom[:, None]).mean(axis=0)
        if np.linalg.norm(score) > 1e-5:
            raise RuntimeError("ELW did not converge sufficiently.")

    denom = 1.0 + h @ lam
    weights = 1.0 / (n * denom)
    if np.any(weights <= 0) or not np.isfinite(weights).all():
        raise RuntimeError("ELW produced invalid weights.")

    weights = weights / weights.sum()
    return weights.astype(np.float64)


def estimate_patternwise_elw_weights(
    x: np.ndarray,
    mask: np.ndarray,
    *,
    balance_features: np.ndarray | BalanceFn | None = None,
    target_moments: TargetMoments | None = None,
    pattern_weight: Literal["frequency", "uniform_patterns"] | Mapping[PatternKey, float] = "frequency",
    max_iter: int = 100,
    tol: float = 1e-9,
    fallback_to_uniform: bool = True,
) -> PatternWeightResult:
    """Estimate pattern-wise ELW sample weights.

    Parameters
    ----------
    x, mask:
        Data and observed-entry mask.
    balance_features:
        Either an ``(n, p)`` array of fully observed balancing features or a
        callable returning ``(G_m, mu_m)`` for each pattern. If omitted, within
        pattern weights are uniform and the method reduces to ordinary CFMI.
    target_moments:
        Target moment vector. If ``balance_features`` is an array and
        ``target_moments`` is omitted, the global mean of ``balance_features``
        is used. A dict keyed by pattern tuples can provide pattern-specific
        targets.
    pattern_weight:
        ``rho_m``. Use empirical frequencies, uniform pattern weights, or a
        user-provided dict.
    """

    x_np = np.asarray(x, dtype=np.float64)
    mask_np = np.asarray(mask, dtype=bool)
    if x_np.ndim != 2 or mask_np.shape != x_np.shape:
        raise ValueError("x and mask must both have shape (n_samples, n_features).")

    n = x_np.shape[0]
    groups = group_by_pattern(mask_np)
    sample_weights = np.zeros(n, dtype=np.float64)
    within: dict[PatternKey, np.ndarray] = {}
    rho_by_pattern: dict[PatternKey, float] = {}

    balance_array: np.ndarray | None
    if balance_features is None or callable(balance_features):
        balance_array = None
    else:
        balance_array = np.asarray(balance_features, dtype=np.float64)
        if balance_array.ndim != 2 or balance_array.shape[0] != n:
            raise ValueError("balance_features must have shape (n_samples, p).")

    if balance_array is not None and target_moments is None:
        default_target = np.nanmean(balance_array, axis=0)
    else:
        default_target = None

    raw_rhos: dict[PatternKey, float] = {}
    for key, indices in groups.items():
        if pattern_weight == "frequency":
            rho = len(indices) / n
        elif pattern_weight == "uniform_patterns":
            rho = 1.0 / len(groups)
        elif isinstance(pattern_weight, Mapping):
            rho = float(pattern_weight[key])
        else:
            raise ValueError(f"Unknown pattern_weight: {pattern_weight}")
        raw_rhos[key] = rho

    rho_sum = sum(raw_rhos.values())
    if rho_sum <= 0:
        raise ValueError("Pattern weights must have positive sum.")
    raw_rhos = {key: value / rho_sum for key, value in raw_rhos.items()}

    for key, indices in groups.items():
        n_m = len(indices)
        omega = np.full(n_m, 1.0 / n_m, dtype=np.float64)

        if balance_features is not None:
            try:
                if callable(balance_features):
                    g_m, mu_m = balance_features(x_np, mask_np, indices, key)
                    g_m = np.asarray(g_m, dtype=np.float64)
                    mu_m = np.asarray(mu_m, dtype=np.float64)
                else:
                    g_m = balance_array[indices]
                    if isinstance(target_moments, Mapping):
                        mu_m = np.asarray(target_moments[key], dtype=np.float64)
                    elif target_moments is not None:
                        mu_m = np.asarray(target_moments, dtype=np.float64)
                    else:
                        mu_m = np.asarray(default_target, dtype=np.float64)

                if g_m.shape[0] <= g_m.shape[1]:
                    raise RuntimeError(
                        "Too few rows for stable ELW calibration in this pattern."
                    )
                omega = empirical_likelihood_weights(
                    g_m,
                    mu_m,
                    max_iter=max_iter,
                    tol=tol,
                )
            except Exception as exc:
                if not fallback_to_uniform:
                    raise
                warnings.warn(
                    f"Using uniform weights for pattern {key}; ELW failed: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        rho = raw_rhos[key]
        sample_weights[indices] = rho * omega
        within[key] = omega
        rho_by_pattern[key] = rho

    sample_weights = sample_weights / sample_weights.sum()
    return PatternWeightResult(
        sample_weights=sample_weights.astype(np.float64),
        within_pattern_weights=within,
        pattern_weights=rho_by_pattern,
        pattern_indices=groups,
    )


def _prepare_incomplete_array(
    x: np.ndarray,
    mask: np.ndarray | None,
    *,
    mean: np.ndarray | None = None,
    std: np.ndarray | None = None,
    standardize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    x_np = np.asarray(x, dtype=np.float32)
    was_1d = x_np.ndim == 1
    if was_1d:
        x_np = x_np[None, :]
    if x_np.ndim != 2:
        raise ValueError("x must have shape (n_samples, n_features) or (n_features,).")

    if mask is None:
        mask_np = np.isfinite(x_np)
    else:
        mask_np = np.asarray(mask, dtype=bool)
        if mask_np.ndim == 1:
            mask_np = mask_np[None, :]
        if mask_np.shape != x_np.shape:
            raise ValueError("mask must have the same shape as x.")

    observed_values = np.where(mask_np, x_np, np.nan)
    if mean is None:
        mean = np.nanmean(observed_values, axis=0).astype(np.float32)
    else:
        mean = np.asarray(mean, dtype=np.float32)
    if std is None:
        std = np.nanstd(observed_values, axis=0).astype(np.float32)
    else:
        std = np.asarray(std, dtype=np.float32)

    if mean.shape[0] != x_np.shape[1] or std.shape[0] != x_np.shape[1]:
        raise ValueError("mean/std length must match x.shape[1].")
    if not np.isfinite(mean).all():
        raise ValueError("Every feature must have at least one observed value.")

    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    x_work = x_np.copy()
    if standardize:
        x_work = (x_work - mean[None, :]) / std[None, :]
    x_work = np.where(mask_np, x_work, 0.0).astype(np.float32)
    return x_work, mask_np, mean, std, was_1d


def random_condition_target_split(
    observed_mask: torch.Tensor,
    *,
    target_probability: float = 0.5,
    ensure_condition_when_possible: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Randomly split observed dimensions into condition and target masks."""

    if not (0.0 < target_probability <= 1.0):
        raise ValueError("target_probability must be in (0, 1].")

    observed = observed_mask.bool()
    condition = torch.zeros_like(observed)
    target = torch.zeros_like(observed)

    for row in range(observed.shape[0]):
        observed_idx = torch.nonzero(observed[row], as_tuple=False).flatten()
        k = int(observed_idx.numel())
        if k == 0:
            continue

        draw = torch.rand(k, device=observed.device) < target_probability
        if not bool(draw.any()):
            draw[torch.randint(k, (1,), device=observed.device)] = True
        if ensure_condition_when_possible and k > 1 and bool(draw.all()):
            draw[torch.randint(k, (1,), device=observed.device)] = False

        target_idx = observed_idx[draw]
        condition_idx = observed_idx[~draw]
        target[row, target_idx] = True
        condition[row, condition_idx] = True

    return condition, target


class PatternWiseELWCFMI:
    """Train and use Pattern-wise ELW-CFMI."""

    def __init__(self, input_dim: int, config: CFMIConfig | None = None) -> None:
        self.input_dim = int(input_dim)
        self.config = config or CFMIConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)

        if self.config.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        self.model = SimpleMLPVelocity(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            hidden_layers=self.config.hidden_layers,
            activation=self.config.activation,
            dropout=self.config.dropout,
        ).to(self.device)

        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.weight_result_: PatternWeightResult | None = None
        self.history_: list[dict[str, float]] = []

    def fit(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        sample_weights: Sequence[float] | None = None,
        balance_features: np.ndarray | BalanceFn | None = None,
        target_moments: TargetMoments | None = None,
        pattern_weight: Literal["frequency", "uniform_patterns"] | Mapping[PatternKey, float] = "frequency",
        verbose: bool = True,
        log_interval: int = 100,
    ) -> list[dict[str, float]]:
        """Fit the shared CFMI velocity field."""

        x_train, mask_np, mean, std, _ = _prepare_incomplete_array(
            x,
            mask,
            standardize=self.config.standardize,
        )
        if x_train.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim={self.input_dim}, but x has {x_train.shape[1]} features."
            )

        self.mean_ = mean
        self.std_ = std

        if sample_weights is None:
            self.weight_result_ = estimate_patternwise_elw_weights(
                np.asarray(x, dtype=np.float64),
                mask_np,
                balance_features=balance_features,
                target_moments=target_moments,
                pattern_weight=pattern_weight,
            )
            weights = self.weight_result_.sample_weights
        else:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if weights.shape[0] != x_train.shape[0]:
                raise ValueError("sample_weights length must match x.shape[0].")
            if np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("sample_weights must be finite and non-negative.")
            weights = weights / weights.sum()
            self.weight_result_ = None

        trainable = mask_np.sum(axis=1) > 0
        if not np.all(trainable):
            warnings.warn(
                "Rows with no observed values cannot form training targets and are skipped.",
                RuntimeWarning,
                stacklevel=2,
            )
        if not bool(trainable.any()):
            raise ValueError("No row has observed values for CFMI training.")

        x_tensor = torch.as_tensor(x_train[trainable], device=self.device)
        mask_tensor = torch.as_tensor(mask_np[trainable], device=self.device)
        weights_tensor = torch.as_tensor(weights[trainable], dtype=torch.float32, device=self.device)
        weights_tensor = weights_tensor / weights_tensor.sum()

        n_train = x_tensor.shape[0]
        batch_size = min(self.config.batch_size, max(1, n_train))
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        self.history_ = []
        self.model.train()

        for step in range(1, self.config.steps + 1):
            batch_idx = torch.randint(n_train, (batch_size,), device=self.device)
            batch_x = x_tensor[batch_idx]
            batch_mask = mask_tensor[batch_idx]
            batch_weights = weights_tensor[batch_idx]

            condition_mask, target_mask = random_condition_target_split(
                batch_mask,
                target_probability=self.config.target_probability,
                ensure_condition_when_possible=self.config.ensure_condition_when_possible,
            )
            target_float = target_mask.float()
            condition_float = condition_mask.float()

            tau = torch.rand(batch_size, 1, device=self.device)
            z = torch.randn_like(batch_x)
            x_target_path = (tau * batch_x + (1.0 - tau) * z) * target_float
            x_condition = batch_x * condition_float
            target_velocity = (batch_x - z) * target_float

            pred_velocity = self.model(
                x_target_path,
                x_condition,
                condition_float,
                target_float,
                tau,
            )
            target_count = target_float.sum(dim=1).clamp_min(1.0)
            per_sample_loss = (((pred_velocity - target_velocity) ** 2) * target_float).sum(dim=1)
            per_sample_loss = per_sample_loss / target_count

            loss = (per_sample_loss * batch_weights * n_train).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if self.config.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            optimizer.step()

            if step == 1 or step == self.config.steps or step % log_interval == 0:
                record = {"step": float(step), "loss": float(loss.detach().cpu())}
                self.history_.append(record)
                if verbose:
                    print(f"step={step:05d} loss={record['loss']:.6f}")

        return self.history_

    @torch.no_grad()
    def impute(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        num_imputations: int = 1,
        ode_steps: int = 50,
        seed: int | None = None,
    ) -> np.ndarray:
        """Impute missing values by integrating the learned velocity field."""

        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Call fit before impute.")
        if num_imputations < 1:
            raise ValueError("num_imputations must be >= 1.")
        if ode_steps < 1:
            raise ValueError("ode_steps must be >= 1.")

        if seed is not None:
            torch.manual_seed(seed)

        x_original = np.asarray(x, dtype=np.float32)
        was_1d_input = x_original.ndim == 1
        if was_1d_input:
            x_original_2d = x_original[None, :]
        else:
            x_original_2d = x_original

        x_work, mask_np, _, _, was_1d = _prepare_incomplete_array(
            x,
            mask,
            mean=self.mean_,
            std=self.std_,
            standardize=self.config.standardize,
        )
        if x_work.shape[1] != self.input_dim:
            raise ValueError(
                f"input_dim={self.input_dim}, but x has {x_work.shape[1]} features."
            )

        observed_mask = torch.as_tensor(mask_np, device=self.device)
        missing_mask = ~observed_mask
        observed_float = observed_mask.float()
        missing_float = missing_mask.float()
        x_observed = torch.as_tensor(x_work, device=self.device)
        x_condition = x_observed * observed_float
        n = x_observed.shape[0]
        dt = 1.0 / ode_steps

        self.model.eval()
        completed: list[np.ndarray] = []
        for _ in range(num_imputations):
            missing_state = torch.randn(n, self.input_dim, device=self.device) * missing_float

            for k in range(ode_steps):
                tau_value = k / ode_steps
                tau = torch.full((n, 1), tau_value, dtype=torch.float32, device=self.device)
                velocity = self.model(
                    missing_state * missing_float,
                    x_condition,
                    observed_float,
                    missing_float,
                    tau,
                )
                missing_state = missing_state + dt * velocity * missing_float

            complete_scaled = x_condition + missing_state * missing_float
            complete_np = complete_scaled.detach().cpu().numpy()
            if self.config.standardize:
                complete_np = complete_np * self.std_[None, :] + self.mean_[None, :]
            complete_np = np.where(mask_np, x_original_2d, complete_np)
            completed.append(complete_np.astype(np.float32))

        result = np.stack(completed, axis=0)
        if was_1d:
            result = result[:, 0, :]
        if num_imputations == 1:
            return result[0]
        return result

    def save(self, path: str | Path) -> None:
        """Save model parameters and standardization statistics."""

        payload = {
            "input_dim": self.input_dim,
            "config": self.config,
            "model_state_dict": self.model.state_dict(),
            "mean": self.mean_,
            "std": self.std_,
        }
        torch.save(payload, Path(path))

    @classmethod
    def load(cls, path: str | Path, map_location: str | torch.device | None = None) -> "PatternWiseELWCFMI":
        """Load a saved model."""

        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
        model = cls(input_dim=payload["input_dim"], config=payload["config"])
        model.model.load_state_dict(payload["model_state_dict"])
        model.mean_ = payload["mean"]
        model.std_ = payload["std"]
        model.model.to(model.device)
        return model


def train_pattern_wise_elw_cfmi(
    x: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    config: CFMIConfig | None = None,
    **fit_kwargs: object,
) -> tuple[PatternWiseELWCFMI, list[dict[str, float]]]:
    """Convenience wrapper returning a fitted model and training history."""

    x_np = np.asarray(x)
    if x_np.ndim != 2:
        raise ValueError("x must have shape (n_samples, n_features).")
    model = PatternWiseELWCFMI(input_dim=x_np.shape[1], config=config)
    history = model.fit(x_np, mask=mask, **fit_kwargs)
    return model, history


def _demo() -> None:
    """Small smoke-test demo with synthetic data."""

    rng = np.random.default_rng(7)
    n, d = 256, 5
    z = rng.normal(size=(n, 2)).astype(np.float32)
    x_full = np.column_stack(
        [
            z[:, 0],
            z[:, 1],
            z[:, 0] + 0.5 * z[:, 1] + rng.normal(scale=0.2, size=n),
            np.sin(z[:, 0]) + rng.normal(scale=0.2, size=n),
            z[:, 1] ** 2 + rng.normal(scale=0.2, size=n),
        ]
    ).astype(np.float32)

    mask = np.ones((n, d), dtype=bool)
    mask[:, 2] = rng.random(n) > (1.0 / (1.0 + np.exp(-x_full[:, 0])))
    mask[:, 4] = rng.random(n) > 0.35
    x_missing = np.where(mask, x_full, np.nan).astype(np.float32)

    # Example balancing features: first two variables, observed for all rows.
    balance = x_full[:, :2]
    config = CFMIConfig(steps=20, batch_size=64, hidden_dim=64, seed=11)
    model, history = train_pattern_wise_elw_cfmi(
        x_missing,
        config=config,
        balance_features=balance,
        verbose=True,
        log_interval=10,
    )
    imputed = model.impute(x_missing[:3], num_imputations=2, ode_steps=10, seed=13)
    print("history:", history[-1])
    print("imputed shape:", imputed.shape)


if __name__ == "__main__":
    _demo()
