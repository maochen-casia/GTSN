"""Held-out map quality and paired action-policy interventions."""
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

GROUPS = {'point': slice(0, 3), 'goal': slice(3, 5), 'future_action': slice(5, 6)}
CHANNELS = ['point_x', 'point_y', 'point_z', 'goal_projected', 'goal_visible', 'future_action']


def oracle_variants():
    result = {'predicted': ()}
    for count in range(1, 4):
        for groups in combinations(GROUPS, count):
            result['oracle_' + '_'.join(groups)] = groups
    return result


def replace_groups(predicted, teacher, groups, zero=False):
    result = predicted.clone()
    for group in groups:
        result[:, GROUPS[group]] = 0 if zero else teacher[:, GROUPS[group]]
    return result


def episode_bootstrap(squared, counts, baseline, routes, seed=20261002, repetitions=5000):
    """Paired, route-stratified episode resampling; frames are not independent."""
    rng = np.random.default_rng(seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(routes == route),
                            (repetitions, int((routes == route).sum())), replace=True)
                            for route in np.unique(routes)], axis=1)
    denominator = counts[draws].sum(1)
    current = np.sqrt(squared[draws].sum(1) / denominator)
    original = np.sqrt(baseline[draws].sum(1) / denominator)
    return {'rmse_95ci_rad': np.quantile(current, [.025, .975]).tolist(),
            'paired_rmse_change_95ci_rad': np.quantile(current - original, [.025, .975]).tolist(),
            'paired_relative_rmse_reduction_95ci': np.quantile(1 - current / original, [.025, .975]).tolist()}


class MapQuality:
    """Foreground-aware scores, zero-map comparisons and metric XYZ errors."""
    def __init__(self, settings, device):
        self.settings = settings
        self.sums = torch.zeros(6, 2, dtype=torch.float64, device=device)
        self.point = torch.zeros(4, dtype=torch.float64, device=device)
        self.heat = torch.zeros(3, 16, dtype=torch.float64, device=device)
        self.thresholds = [.1, .25, .5]
        self.confusion = torch.zeros(3, 3, 3, dtype=torch.float64, device=device)
        self.histograms = torch.zeros(3, 2, 512, dtype=torch.float64, device=device)
        self.pixels = 0
        self.frames = 0

    @torch.no_grad()
    def update(self, predicted, teacher, depth):
        p, t = predicted.float(), teacher.float()
        error = (p - t).square()
        self.sums[:, 0] += error.sum((0, 2, 3)).double()
        self.sums[:, 1] += t.square().sum((0, 2, 3)).double()
        self.pixels += len(t) * t.shape[-2] * t.shape[-1]
        self.frames += len(t)
        z = F.interpolate(depth[:, None].float(), t.shape[-2:], mode='nearest-exact')[:, 0]
        observed = torch.isfinite(z) & (z >= self.settings.near_m) & (z <= self.settings.far_m)
        scale = t.new_tensor(self.settings.point_scale_m)[None, :, None, None]
        distance = ((p[:, :3] - t[:, :3]) * scale).square().sum(1).sqrt()
        self.point += torch.stack([(distance * observed).sum(),
                                   (distance.square() * observed).sum(), observed.sum(),
                                   (observed & (distance <= .02)).sum()]).double()
        width = t.shape[-1]
        for i in range(3):
            actual, estimated = t[:, 3+i], p[:, 3+i]
            foreground = actual >= .1
            e = error[:, 3+i]
            active = actual.flatten(1).amax(1) >= .1
            gt_index, pd_index = actual.flatten(1).argmax(1), estimated.flatten(1).argmax(1)
            delta = ((gt_index // width - pd_index // width).float().square() +
                     (gt_index % width - pd_index % width).float().square()).sqrt()
            predicted_active = estimated.flatten(1).amax(1) >= .1
            values = [e.sum(), actual.square().sum(), (e * foreground).sum(), foreground.sum(),
                      (e * ~foreground).sum(), (~foreground).sum(),
                      (actual.square() * foreground).sum(), (estimated * foreground).sum(),
                      (actual * foreground).sum(), (delta * active).sum(), active.sum(),
                      (active & (delta <= 5)).sum(), (active & predicted_active).sum(),
                      predicted_active.sum(), estimated.sum(), actual.sum()]
            self.heat[i] += torch.stack(values).double()
            for j, threshold in enumerate(self.thresholds):
                truth, prediction = actual >= threshold, estimated >= threshold
                self.confusion[i, j] += torch.stack([(truth & prediction).sum(),
                                                     (~truth & prediction).sum(),
                                                     (truth & ~prediction).sum()]).double()
            bins = (estimated.clamp(0, 1) * 511).long()
            self.histograms[i, 0] += torch.bincount(bins[foreground], minlength=512).double()
            self.histograms[i, 1] += torch.bincount(bins[~foreground], minlength=512).double()

    def result(self):
        sums, point, heat, confusion, histograms = [x.cpu().numpy() for x in
            (self.sums, self.point, self.heat, self.confusion, self.histograms)]
        def divide(a, b):
            return float(a / b) if b else None
        result = {'frames': self.frames, 'foreground_threshold': .1,
                  'pixel_rmse': dict(zip(CHANNELS, np.sqrt(sums[:, 0] / self.pixels).tolist())),
                  'zero_map_pixel_rmse': dict(zip(CHANNELS, np.sqrt(sums[:, 1] / self.pixels).tolist())),
                  'point': {'valid_pixels': int(point[2]), 'mean_euclidean_error_m': divide(point[0], point[2]),
                            'rms_euclidean_error_m': np.sqrt(divide(point[1], point[2])),
                            'fraction_within_2cm': divide(point[3], point[2]),
                            'note': 'Metric error relative to normalized/clipped teacher XYZ, on valid depth pixels.'},
                  'heatmaps': {}}
        for i, name in enumerate(CHANNELS[3:]):
            h = heat[i]
            positive, negative = histograms[i, :, ::-1]
            tp, fp = np.cumsum(positive), np.cumsum(negative)
            precision = tp / np.maximum(tp + fp, 1)
            ap = divide((precision * positive).sum(), positive.sum())
            result['heatmaps'][name] = {
                'foreground_fraction': divide(h[3], self.pixels),
                'foreground_rmse': np.sqrt(divide(h[2], h[3])) if h[3] else None,
                'zero_foreground_rmse': np.sqrt(divide(h[6], h[3])) if h[3] else None,
                'background_rmse': np.sqrt(divide(h[4], h[5])) if h[5] else None,
                'foreground_mean_prediction': divide(h[7], h[3]),
                'foreground_mean_target': divide(h[8], h[3]),
                'mean_prediction': divide(h[14], self.pixels), 'mean_target': divide(h[15], self.pixels),
                'approximate_pixel_average_precision': ap, 'ap_histogram_bins': 512,
                'active_frames': int(h[10]), 'mean_peak_error_pixels': divide(h[9], h[10]),
                'peak_within_5px_rate': divide(h[11], h[10]),
                'active_frame_detection_recall': divide(h[12], h[10]),
                'active_frame_detection_precision': divide(h[12], h[13]),
                'threshold_scores': {}}
            for j, threshold in enumerate(self.thresholds):
                a, b, c = confusion[i, j]
                result['heatmaps'][name]['threshold_scores'][str(threshold)] = {
                    'precision': divide(a, a+b), 'recall': divide(a, a+c),
                    'dice': divide(2*a, 2*a+b+c), 'iou': divide(a, a+b+c)}
        return result
