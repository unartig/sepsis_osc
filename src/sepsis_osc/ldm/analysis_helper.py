import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from tbparse import SummaryReader

from sepsis_osc.ldm.checkpoint_utils import load_checkpoint
from sepsis_osc.ldm.data_loading import get_data_sets_online, get_raw_data
from sepsis_osc.ldm.latent_dynamics_model import LatentDynamicsModel
from sepsis_osc.ldm.model_structs import AuxLosses, LoadingConfig, LossesConfig
from sepsis_osc.ldm.train_online import process_val_epoch
from sepsis_osc.utils.config import CV_FOLDS, CV_REPETITIONS, HIGH_SOFA_THRESH, jax_random_seed
from sepsis_osc.utils.logger import setup_logging

setup_logging("info")
logger = logging.getLogger(__name__)

LOSS_CONF = {
    "lambda_sep3": 300.0,
    "lambda_inf": 1.0,
    "lambda_sofa_classification": 2000.0,
    "lambda_spreading": 6e-3,
    "lambda_boundary": 30.0,
    "lambda_recon": 2.5,
}


@dataclass
class RunResult:
    """Everything from loading + inferring one (scenario, rep, fold) split."""

    scenario: str
    rep: int
    fold: int
    model: LatentDynamicsModel
    best_epoch: int
    tb_df: pd.DataFrame
    hparams: pd.DataFrame
    metrics: AuxLosses  # already converted via .to_np()
    shared_data: dict

    @property
    def data(self) -> tuple:
        return self.shared_data[(self.rep, self.fold)]

    @property
    def mask(self) -> np.ndarray:
        return self.data[2]

    @property
    def beta(self) -> np.ndarray:
        return self.metrics.beta[self.mask]

    @property
    def sigma(self) -> np.ndarray:
        return self.metrics.sigma[self.mask]


@dataclass
class AnalysisResult:
    """Small derived summaries for one split — cached, not the raw arrays."""

    run: RunResult
    perf: dict
    subgroup_stats: pd.DataFrame
    correlation: pd.DataFrame
    recon_mse: np.ndarray
    recon_rmse: np.ndarray
    recon_r2: np.ndarray
    recon_pr: np.ndarray

    @property
    def scenario(self) -> str:
        return self.run.scenario

    @property
    def rep(self) -> int:
        return self.run.rep

    @property
    def fold(self) -> float:
        return self.run.fold


@dataclass
class Scenario:
    """All splits for one scenario, plus aggregated summaries."""

    name: str
    results: list[AnalysisResult]

    @property
    def perf_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.perf | {"rep": r.rep, "fold": r.fold} for r in self.results])

    @property
    def subgroup_stats_mean(self) -> pd.DataFrame | None:
        all_stats = [r.subgroup_stats for r in self.results]
        if not all_stats:
            return None
        return pd.concat(all_stats).groupby(["comparison", "coord"])["cohens_d"].agg(["mean", "std"]).reset_index()

    @property
    def corr_mean(self) -> pd.DataFrame | None:
        corrs = [r.correlation for r in self.results]
        return pd.concat(corrs).groupby(level=0).mean() if corrs else None

    @property
    def decoder_mse_mean(self) -> np.ndarray | None:
        mses = [r.recon_mse for r in self.results]
        return np.stack(mses).mean(axis=0) if mses else None

    def representative(self) -> AnalysisResult:
        """Split closest to the validation mean (AUROC, AUPRC)"""
        _rows = []
        for r in self.results:
            _tb_df = r.run.tb_df
            _auprc = _tb_df.query("tag == 'sepsis_metrics/AUPRC_pred_sep'")
            _auroc = _tb_df.query("tag == 'sepsis_metrics/AUROC_pred_sep'")
            _rows.append({"rep": r.rep, "fold": r.fold, "auprc": _auprc.max().value, "auroc": _auroc.max().value})
        _df = pd.DataFrame(_rows)
        _mean_auroc, _mean_auprc = _df["auroc"].mean(), _df["auprc"].mean()
        _dist = np.sqrt((_df["auroc"] - _mean_auroc) ** 2 + (_df["auprc"] - _mean_auprc) ** 2)
        return self.results[_dist.idxmin()]


def get_data_and_features(yaib_data_dir: str) -> tuple[dict[str, pl.DataFrame], pd.Series]:
    raw_data = get_raw_data(_data_dir=Path(yaib_data_dir))
    _col_names = raw_data["test"]["FEATURES"].columns
    features = pd.Series(
        {
            col: i - 2
            for i, col in enumerate(_col_names)
            if col not in ("stay_id", "time") and not col.startswith("MissingIndicator_")
        }
    )
    print(f"Features: {len(features)}\n{list(features.index)}")
    return raw_data, features


def pre_load_all_data_parallel(
    sequence_files: str,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data_store = {}

    def _load_split(rep: int, fold: int) -> tuple[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] | None:
        try:
            logger.info(f"Loading data rep={rep} fold={fold}")
            data = get_data_sets_online(
                swapaxes_y=(1, 2, 0),
                dtype=jnp.float32,
                cv_repetitions=CV_REPETITIONS,
                repetition_index=rep,
                cv_folds=CV_FOLDS,
                fold_index=fold,
                sequence_files=sequence_files,
            )
            *_, test_x, test_y, test_m = data
            test_m = test_m.astype(bool)
            return (rep, fold), (test_x[None], test_y[None], test_m[None])
        except Exception:
            logger.exception(f"Failed rep={rep} fold={fold}")
            return None

    tasks = [(rep, fold) for rep in range(CV_REPETITIONS) for fold in range(CV_FOLDS)]

    with ThreadPoolExecutor(max_workers=2) as pool:
        for item in pool.map(lambda p: _load_split(*p), tasks):
            if item is not None:
                key_pair, arrays = item
                data_store[key_pair] = arrays

    return data_store


def get_run_dir(scenario: str, rep: int, fold: int, run_base: str) -> str:
    run_name = f"rep{rep:02d}_fold{fold:02d}"
    return f"{run_base}/{scenario}/{run_name}"


def load_tb_df(
    scenario: str, rep_fold: tuple[int, int], run_base: str
) -> tuple[pd.DataFrame, int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rep, fold = rep_fold
    tb = SummaryReader(get_run_dir(scenario, rep, fold, run_base))
    tb_df = tb.scalars
    hparams = tb.hparams.T
    hparams.columns = hparams.loc["tag"]
    hparams = hparams.drop("tag")

    auprc_df = tb_df[tb_df["tag"] == "sepsis_metrics/AUPRC_pred_sep"]
    auroc_df = tb_df[tb_df["tag"] == "sepsis_metrics/AUROC_pred_sep"]

    prc_epoch = auprc_df.loc[auprc_df["value"].idxmax(), "step"]
    roc_epoch = auroc_df.loc[auroc_df["value"].idxmax(), "step"]
    best_epoch = round((prc_epoch + roc_epoch) / 2)

    logger.info(
        f"[{scenario} rep{rep:02d}_fold{fold:02d}] best_epoch={best_epoch} "
        f"AUPRC={auprc_df['value'].max():.3f}@{prc_epoch} "
        f"AUROC={auroc_df['value'].max():.3f}@{roc_epoch}"
    )
    return tb_df, best_epoch, hparams, auroc_df, auprc_df


def load_model(scenario: str, rep: int, fold: int, epoch: int, run_base: str) -> LatentDynamicsModel:
    load_conf = LoadingConfig(from_dir=get_run_dir(scenario, rep, fold, run_base), epoch=epoch)
    model, _ = load_checkpoint(load_conf.from_dir + "/checkpoints", load_conf.epoch, None)
    model = eqx.nn.inference_mode(model)
    return model


def load_data(rep: int, fold: int, shared_data: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pre-loaded split from shared_data cache (no disk I/O)."""
    return shared_data[(rep, fold)]


def run_inference(model: LatentDynamicsModel, test_x: np.ndarray, test_y: np.ndarray, test_m: np.ndarray) -> AuxLosses:
    key = jr.PRNGKey(jax_random_seed)
    loss_conf = LossesConfig(**LOSS_CONF)  # ty:ignore[invalid-argument-type]
    return process_val_epoch(
        model,
        x_data=test_x,
        y_data=test_y,
        mask_data=test_m,
        step=jnp.array(1_000_000, dtype=jnp.int32),
        key=key,
        loss_params=loss_conf,
    )


def extract_latents(metrics: AuxLosses, mask: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Returns (beta_ts, sigma_ts) as 1-D arrays over masked timesteps."""
    beta_ts = np.asarray(metrics.beta)[mask]
    sigma_ts = np.asarray(metrics.sigma)[mask]
    return beta_ts, sigma_ts


def extract_labels(test_y: jnp.ndarray, mask: jnp.ndarray) -> tuple[jnp.ndarray, ...]:
    """Returns (sep, inf, sofa, peak_sofa_ts) over masked timesteps."""
    sofa_3d = np.asarray(test_y[..., 0])
    inf_3d = np.asarray(test_y[..., 1])
    sep_3d = np.asarray(test_y[..., 2])
    mask_3d = np.asarray(mask)

    sep = sep_3d[mask_3d]
    inf_ = inf_3d[mask_3d]
    sofa = sofa_3d[mask_3d]

    sofa_masked = np.where(mask_3d, sofa_3d, -np.inf)
    peak_per_pat = sofa_masked.max(axis=-1)  # (batch, n_pat)
    peak_expanded = np.broadcast_to(peak_per_pat[..., None], sofa_3d.shape)
    peak_sofa = peak_expanded[mask_3d]

    return sep, inf_, sofa, peak_sofa


def extract_inputs(test_x: jnp.ndarray, mask: jnp.ndarray, features: pd.DataFrame) -> pd.DataFrame:
    """Returns DataFrame of input features over masked timesteps."""
    return pd.DataFrame(
        np.asarray(test_x[..., features.values])[mask],
        columns=features.index,
    )


def compute_performance(metrics: AuxLosses, test_y: jnp.ndarray, mask: jnp.ndarray) -> dict[str, float]:
    true_sep = np.asarray(test_y[..., 2] == 1.0)[mask]
    true_inf = np.asarray(test_y[..., 1])[mask]
    true_sofa = np.asarray(test_y[..., 0])
    true_sofa_d2 = np.concatenate(
        [
            np.zeros(test_y.shape[:-2])[..., None],
            np.asarray(np.diff(test_y[..., 0], axis=-1) > 0),
        ],
        axis=-1,
    )[mask]

    pred_sep = np.asarray(metrics.sep3_risk)[mask]
    pred_inf = np.asarray(metrics.susp_inf_p)[mask]
    pred_sofa_d2 = np.asarray(metrics.sofa_d2_risk)[mask]
    pred_sofa = np.asarray(metrics.hists_sofa_score)

    return {
        "auroc_sep3": roc_auc_score(true_sep, pred_sep),
        "auprc_sep3": average_precision_score(true_sep, pred_sep),
        "auroc_sofa_d2": roc_auc_score(true_sofa_d2, pred_sofa_d2),
        "auprc_sofa_d2": average_precision_score(true_sofa_d2, pred_sofa_d2),
        "auroc_inf": roc_auc_score(true_inf > 0, pred_inf),
        "auprc_inf": average_precision_score(true_inf > 0, pred_inf),
        "auroc_sep3_sofa_only": roc_auc_score(true_sep, pred_sofa_d2),
        "auprc_sep3_sofa_only": average_precision_score(true_sep, pred_sofa_d2),
        "auroc_sep3_inf_only": roc_auc_score(true_sep, pred_inf),
        "auprc_sep3_inf_only": average_precision_score(true_sep, pred_inf),
        "auroc_sep3_sofa_only_gt": roc_auc_score(true_sep, pred_sofa_d2),
        "auprc_sep3_sofa_only_gt": average_precision_score(true_sep, pred_sofa_d2),
        "auroc_sep3_inf_only_gt": roc_auc_score(true_sep, true_inf),
        "auprc_sep3_inf_only_gt": average_precision_score(true_sep, true_inf),
        "rmse_sofa": np.sqrt(np.mean((true_sofa - pred_sofa) ** 2)),
    }


def compute_alignment(others: dict, features_df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({k: features_df.apply(lambda c: np.corrcoef(v, c)[0, 1]) for k, v in others.items()})


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (a.mean() - b.mean()) / pooled if pooled > 0 else jnp.nan


def compute_subgroup_stats(
    beta_ts: np.ndarray, sigma_ts: np.ndarray, sep: np.ndarray, inf_: np.ndarray, peak_sofa: np.ndarray
) -> pd.DataFrame:
    comparisons = [
        ("Sepsis", sep == 1.0, "No sepsis", sep != 1.0),
        ("Infection", inf_ > 0, "No infection", inf_ == 0),
        ("High SOFA", peak_sofa >= HIGH_SOFA_THRESH, "Low SOFA", peak_sofa < HIGH_SOFA_THRESH),
    ]
    rows = []
    for la, ma, lb, mb in comparisons:
        for coord, arr in [(r"$\beta$", beta_ts), (r"$\sigma$", sigma_ts)]:
            rows.append(
                {
                    "comparison": f"{la} vs {lb}",
                    "coord": coord,
                    f"{la}_mean": arr[ma].mean(),
                    f"{la}_std": arr[ma].std(),
                    f"{lb}_mean": arr[mb].mean(),
                    f"{lb}_std": arr[mb].std(),
                    "cohens_d": _cohens_d(arr[ma], arr[mb]),
                    "n_a": ma.sum(),
                    "n_b": mb.sum(),
                }
            )
    return pd.DataFrame(rows)


def compute_recon(model: LatentDynamicsModel, beta: jnp.ndarray, sigma: jnp.ndarray) -> np.ndarray:
    bs = np.concatenate([beta[..., None], sigma[..., None]], axis=-1)
    # run decoder: vmap over (batch, patients, time)
    recon_np = np.asarray(jax.vmap(jax.vmap(jax.vmap(model.decoder)))(jnp.asarray(bs)))
    return recon_np


def compute_decoder_performance(
    recon: np.ndarray, test_x: np.ndarray, mask: np.ndarray, n_features: int = 52
) -> tuple[np.ndarray, ...]:
    mask = mask.astype(bool)
    recon_mse = np.square(recon - test_x[..., :n_features]).mean(axis=(0, 1, 2), where=mask[..., None])

    recon_rmse = np.sqrt(recon_mse)

    feature_mean_broadcast = np.asarray(test_x[..., :n_features]).mean(
        axis=(0, 1, 2), where=mask[..., None], keepdims=True
    )
    feature_var = np.mean(
        np.square(test_x[..., :n_features] - feature_mean_broadcast), axis=(0, 1, 2), where=mask[..., None]
    )
    recon_r2 = 1 - (recon_mse / feature_var)

    recon_pr = np.zeros(n_features)
    for f in range(n_features):
        valid_target = test_x[mask, f]
        valid_recon = recon[mask, f]

        r_val, _ = stats.pearsonr(valid_target, valid_recon)
        recon_pr[f] = r_val
    return recon_mse, recon_r2, recon_rmse, recon_pr


def performance_table(perf_dict: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """perf_dict: {scenario: {metric: (mean, std)}}"""
    metrics = ["auroc_sep3", "auprc_sep3", "auroc_sofa_d2", "auprc_sofa_d2", "auroc_inf", "auprc_inf"]
    rows = []
    for scenario, d in perf_dict.items():
        row = {"scenario": scenario}
        for m in metrics:
            if m in d:
                mean, std = d[m]
                row[m] = f"{mean * 100:.2f} ± {std * 100:.2f}"
            else:
                row[m] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("scenario")


def run_split(scenario: str, rep: int, fold: int, shared_data: dict, run_base: str) -> RunResult | None:
    """Load tb_df/model/checkpoint + run inference"""
    try:
        tb_df, best_epoch, hparams, _, _ = load_tb_df(scenario, (rep, fold), run_base)
        model = load_model(scenario, rep, fold, best_epoch, run_base)
        test_x, test_y, test_m = load_data(rep, fold, shared_data)
        metrics = run_inference(model, test_x, test_y, test_m).to_np()
        return RunResult(scenario, rep, fold, model, best_epoch, tb_df, hparams, metrics, shared_data)
    except Exception:
        logger.exception(f"run_split failed {scenario} rep={rep} fold={fold}")
        return None


def analyse_run(run: RunResult, features: pd.DataFrame) -> AnalysisResult | None:
    """Compute perf/subgroup/correlation/recon stats from a RunResult."""
    try:
        test_x, test_y, test_m = run.data
        sep, inf_, _, peak = extract_labels(test_y, test_m)
        x_df = extract_inputs(test_x, test_m, features=features)

        perf = compute_performance(run.metrics, test_y, test_m)
        substat = compute_subgroup_stats(run.beta, run.sigma, np.asarray(sep), np.asarray(inf_), np.asarray(peak))
        corr = compute_alignment({"beta": run.beta, "sigma": run.sigma}, x_df)

        recon = compute_recon(run.model, run.metrics.beta, run.metrics.sigma)
        recon_mse, recon_r2, recon_rmse, recon_pr = compute_decoder_performance(recon, test_x, test_m)

        return AnalysisResult(
            run=run,
            perf=perf,
            subgroup_stats=substat,
            correlation=corr,
            recon_mse=recon_mse,
            recon_r2=recon_r2,
            recon_rmse=recon_rmse,
            recon_pr=recon_pr,
        )
    except Exception:
        logger.exception(f"analyse failed {run.scenario} rep={run.rep} fold={run.fold}")
        return None


def build_scenario(
    scenario: str, repetitions: int, folds: int, run_base: str, shared_data: str, features: pd.DataFrame
) -> Scenario:
    splits = [(rep, fold) for rep in range(repetitions) for fold in range(folds)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        runs = [
            r
            for r in pool.map(lambda rf: run_split(scenario, *rf, shared_data, run_base), splits)
            if r is not None
        ]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [r for r in pool.map(analyse_run, runs, [features] * len(runs)) if r is not None]
    return Scenario(scenario, results)

