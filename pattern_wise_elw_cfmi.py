"""Pattern-wise ELW-CFMI with a simple MLP velocity field.

The implementation follows the pseudo-code in
``pattern_wise_elw_cfmi_reformatted.pdf``:

1. Split samples by observed-missing pattern.
2. Use ``pattern_wise_elw.py`` to compute sample-level ELW weights.
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
from typing import Literal
from typing import Mapping
from typing import Sequence
import warnings

import numpy as np
import torch
from torch import nn

from pattern_wise_elw import Pattern
from pattern_wise_elw import PatternWiseELWResult
from pattern_wise_elw import estimate_propensity_by_pattern
from pattern_wise_elw import pattern_wise_elw_weights


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
        self.elw_result_: PatternWiseELWResult | None = None
        self.history_: list[dict[str, float]] = []

    def fit(
        self,
        x: np.ndarray,
        mask: np.ndarray | None = None,
        *,
        sample_weights: Sequence[float] | None = None,
        propensity_by_pattern: Mapping[Pattern, np.ndarray] | None = None,
        estimate_propensity: bool = False,
        propensity_kwargs: Mapping[str, object] | None = None,
        rho_by_pattern: Mapping[Pattern, float] | None = None,
        elw_clip: tuple[float, float] | None = (1e-6, 1.0 - 1e-6),
        elw_tol: float = 1e-12,
        elw_max_iter: int = 200,
        verbose: bool = True,
        log_interval: int = 100,
    ) -> list[dict[str, float]]:
        """Fit the shared CFMI velocity field.

        ELW weights are delegated to ``pattern_wise_elw.py``. Pass externally
        estimated ``propensity_by_pattern`` to use ``pattern_wise_elw_weights``
        directly, or set ``estimate_propensity=True`` to use its baseline
        ``estimate_propensity_by_pattern`` helper. If neither is supplied,
        uniform sample weights are used.
        """

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

        if sample_weights is not None:
            weights = np.asarray(sample_weights, dtype=np.float64)
            if weights.shape[0] != x_train.shape[0]:
                raise ValueError("sample_weights length must match x.shape[0].")
            if np.any(weights < 0) or not np.isfinite(weights).all():
                raise ValueError("sample_weights must be finite and non-negative.")
            weights = weights / weights.sum()
            self.elw_result_ = None
        elif propensity_by_pattern is not None or estimate_propensity:
            if propensity_by_pattern is not None and estimate_propensity:
                raise ValueError(
                    "Pass either propensity_by_pattern or estimate_propensity=True, not both."
                )
            if estimate_propensity:
                kwargs = dict(propensity_kwargs or {})
                propensity_by_pattern = estimate_propensity_by_pattern(
                    np.asarray(x, dtype=np.float64),
                    observed_mask=mask_np,
                    **kwargs,
                )
            self.elw_result_ = pattern_wise_elw_weights(
                mask_np,
                propensity_by_pattern,
                rho_by_pattern=rho_by_pattern,
                clip=elw_clip,
                tol=elw_tol,
                max_iter=elw_max_iter,
            )
            weights = np.asarray(self.elw_result_.training_weights, dtype=np.float64)
            weights = weights / weights.sum()
        else:
            weights = np.full(x_train.shape[0], 1.0 / x_train.shape[0], dtype=np.float64)
            self.elw_result_ = None

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

    config = CFMIConfig(steps=20, batch_size=64, hidden_dim=64, seed=11)
    model, history = train_pattern_wise_elw_cfmi(
        x_missing,
        config=config,
        estimate_propensity=True,
        propensity_kwargs={"cross_fit": 3, "random_state": 11},
        verbose=True,
        log_interval=10,
    )
    imputed = model.impute(x_missing[:3], num_imputations=2, ode_steps=10, seed=13)
    print("history:", history[-1])
    print("imputed shape:", imputed.shape)


if __name__ == "__main__":
    _demo()
