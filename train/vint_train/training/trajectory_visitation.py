"""Soft visitation targets and metrics from existing local XY waypoints only."""

import math

import torch
from torch.nn import functional as F
from vint_train.data.trajectory_regions import metric_visitation_histograms


def visitation_targets(batch, config, device=None):
    """Return [B,S,R] normalized histograms, valid samples and waypoint counts.

    Construct on CPU in float64 in both denominator and loss paths. No geometry
    labels are loaded; only the already-existing action target/mask are read.
    Exclude stationary points (undefined bearing), negative/mined goals and all
    other action-invalid samples. Repeated nonstationary points retain their
    time weight. Unknown, occupied and free are not target classes.
    """
    definition = config["spatial_bottleneck"]["discretization"]
    actions = batch[2].detach().to(device="cpu", dtype=torch.float64)
    selected = batch[6].detach().to(device="cpu") > 0
    if actions.ndim != 3 or actions.shape[-1] != 2 or selected.shape != (len(actions),):
        raise ValueError("SPATIAL_BOTTLENECK expects XY action targets [B,H,2] and action_mask [B]")
    if not torch.isfinite(actions[selected]).all():
        raise ValueError("SPATIAL_BOTTLENECK valid action targets must be finite")
    scale = float(config["action_scale_m"])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("SPATIAL_BOTTLENECK action_scale_m must be finite and positive")
    # This trainer supplies a single manifest scale and dataset_index=0. Fail
    # instead of silently applying that scale to a heterogeneous dataset batch.
    if batch[5] is not None and (batch[5] != 0).any():
        raise ValueError("SPATIAL_BOTTLENECK expects the single-manifest dataset_index=0 scale contract")
    points = torch.where(selected[:, None, None], actions, torch.zeros_like(actions)) * scale
    histogram, counts = metric_visitation_histograms(points.numpy(), definition)
    target, count = torch.from_numpy(histogram).float(), torch.from_numpy(counts)
    valid = selected & (count > 0)
    if device is not None:
        target, valid, count = target.to(device), valid.to(device), count.to(device)
    return target, valid, count


def visitation_loss_sum(logits, target, valid):
    if logits.shape != target.shape:
        raise ValueError("SPATIAL_BOTTLENECK visitation logits and targets must have the same [B,S,R] shape")
    if not valid.any():
        return logits.new_zeros((), dtype=torch.float32)
    log_probability = F.log_softmax(logits[valid].float().flatten(1), dim=-1)
    return -(target[valid].float().flatten(1) * log_probability).sum()


class VisitationMetrics:
    """Dataset-level sums, not averages of batch means."""

    def __init__(self, head_trained=True, prior_probability=None, token_permutation=False):
        self.head_trained = head_trained
        self.samples = self.valid_samples = self.waypoints = 0
        self.scored_samples = self.nonfinite_predictions = 0
        self.sums = {key: 0.0 for key in ("cross_entropy", "target_entropy", "l1", "mass_on_visited_bins", "top1_visited")}
        self.target_mass = self.predicted_mass = None
        self.prior_probability = prior_probability
        self.prior = VisitationMetrics() if prior_probability is not None else None
        self.permuted = VisitationMetrics() if token_permutation else None
        self.paired_original = VisitationMetrics() if token_permutation else None
        self.permutation_skipped_samples = self.condition_samples = 0
        self.condition_delta_sum = 0.

    @torch.no_grad()
    def update(self, logits, target, valid, counts):
        if self.prior is not None:
            prior_logits = target.new_tensor(self.prior_probability).log().reshape(1, *target.shape[1:])
            self.prior.update(prior_logits.expand_as(target), target, valid, counts)
        self.samples += len(valid)
        self.valid_samples += int(valid.sum())
        self.waypoints += int(counts[valid].sum())
        if not self.head_trained or not valid.any():
            return
        selected_logits = logits[valid].float().flatten(1)
        finite = torch.isfinite(selected_logits).all(dim=1)
        self.nonfinite_predictions += int((~finite).sum())
        self.scored_samples += int(finite.sum())
        if not finite.any():
            return
        logp = F.log_softmax(selected_logits[finite], dim=-1)
        p, q = logp.exp(), target[valid][finite].float().flatten(1)
        self.sums["cross_entropy"] += float(-(q * logp).sum())
        self.sums["target_entropy"] += float(-(q * q.clamp_min(1.e-12).log()).sum())
        self.sums["l1"] += float((p - q).abs().sum())
        self.sums["mass_on_visited_bins"] += float((p * (q > 0)).sum())
        self.sums["top1_visited"] += float((q.gather(1, p.argmax(dim=1, keepdim=True)) > 0).sum())
        q_sum, p_sum = q.sum(dim=0).cpu(), p.sum(dim=0).cpu()
        self.target_mass = q_sum if self.target_mass is None else self.target_mass + q_sum
        self.predicted_mass = p_sum if self.predicted_mass is None else self.predicted_mass + p_sum

    @torch.no_grad()
    def update_token_permutation(self, model, encoded, target, valid, counts):
        """Swap entire token sets across valid samples, holding c and labels fixed.

        This is an eval-only reliance diagnostic, not a geometry/causality test.
        Paired original metrics use exactly the same scored subset.
        """
        if self.permuted is None or encoded["spatial_tokens"] is None:
            return
        if model.training:
            raise ValueError("Token permutation diagnostics require model.eval()")
        if int(valid.sum()) < 2:
            self.permutation_skipped_samples += int(valid.sum())
            return
        module = model.vision_encoder.spatial_bottleneck
        tokens = encoded["spatial_tokens"][valid].roll(1, dims=0)
        condition = encoded["base_condition"][valid]
        paired_valid = torch.ones(len(tokens), device=valid.device, dtype=torch.bool)
        logits = module.predict_visitation(tokens, condition)
        self.paired_original.update(encoded["visitation_logits"][valid], target[valid], paired_valid, counts[valid])
        self.permuted.update(logits, target[valid], paired_valid, counts[valid])
        if model.vision_encoder.spatial_bottleneck_mode == "bottleneck":
            difference = (module.fuse(condition, tokens) - encoded["condition"][valid]).float().norm(dim=-1)
            finite = torch.isfinite(difference)
            self.condition_samples += int(finite.sum())
            self.condition_delta_sum += float(difference[finite].sum())

    def report(self):
        count = self.scored_samples
        means = {key: value / count if count else None for key, value in self.sums.items()}
        difference = means["cross_entropy"] - means["target_entropy"] if count else float("nan")
        means["kl_target_to_prediction"] = max(0.0, difference) if math.isfinite(difference) else None
        result = {"head_trained": self.head_trained, "samples": self.samples,
                "valid_samples": self.valid_samples, "valid_waypoints": self.waypoints,
                "scored_samples": self.scored_samples, "nonfinite_prediction_samples": self.nonfinite_predictions,
                **means, "mean_target_mass_flat": (self.target_mass / count).tolist() if count else None,
                "mean_predicted_mass_flat": (self.predicted_mass / count).tolist() if count else None}
        if self.prior is not None:
            result["train_prior"] = self.prior.report()
        if self.permuted is not None:
            original, permuted = self.paired_original.report(), self.permuted.report()
            comparable = (original["scored_samples"] > 0 and original["nonfinite_prediction_samples"] == 0
                          and permuted["nonfinite_prediction_samples"] == 0)
            result["token_permutation"] = {
                "paired_original": original, "permuted": permuted,
                "cross_entropy_increase": (permuted["cross_entropy"] - original["cross_entropy"]) if comparable else None,
                "skipped_singleton_samples": self.permutation_skipped_samples,
                "mean_condition_l2_change": self.condition_delta_sum / self.condition_samples if self.condition_samples else None,
                "condition_scored_samples": self.condition_samples}
        return result
