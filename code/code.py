from pathlib import Path
import numpy as np

import random
import rasterio

import sys

# Import the customized local pysocialforce package instead of any
# environment-level package with the same name.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pysocialforce as psf
from pysocialforce.utils import stateutils

import time
from scipy.spatial import cKDTree
import geopandas as gpd
import pickle
import scipy.io as sio
import json
from numba import njit
from types import MethodType

from pysocialforce import forces as force_module


SIMULATION_STEPS = 10000
# Write all three Monte Carlo scenarios to the shared Results and images
# directories and distinguish them by filename suffixes.
RESULTS_DIR = PROJECT_DIR.joinpath("results")
IMAGES_DIR = PROJECT_DIR.joinpath("animation")
DATA_DIR = PROJECT_DIR.joinpath("data")

# A fixed seed makes the pre-movement-time samples reproducible.
MC_RANDOM_SEED = 20260727
# Number of Monte Carlo realizations added per convergence update.
MC_BATCH_SIZE = 250
# Do not test convergence before this number of samples is reached.
MC_MIN_SAMPLES = 7000
# Stop at this upper bound if convergence is not achieved.
MC_MAX_SAMPLES = 100000
# Maximum relative change in summary statistics between consecutive batches.
MC_RELATIVE_CHANGE_TOLERANCE = 0.01
# Compare cumulative curves at 10-minute intervals over a four-hour window.
MC_CURVE_INTERVAL_MINUTES = 10
MC_CURVE_TIME_MAX_MINUTES = 4 * 60
# Require several consecutive stable batches to avoid accidental convergence.
MC_STABLE_BATCHES_REQUIRED = 3

# Keep shelter selection and speed shuffling reproducible and independent of
# the pre-movement-time Monte Carlo stream.
FIXED_RANDOM_SEED = 10

# Use a separate seed for speed sampling. Convergence requires the maximum
# pointwise change in each mean ordered-speed curve to be at most 0.01 m/s.
SPEED_MC_RANDOM_SEED = 20260728
SPEED_MC_CURVE_DIFFERENCE_TOLERANCE = 0.01


def sample_converged_premovement_times(
    household_sizes,
    alpha_time,
    lam,
    scale_response,
):
    """Estimate household pre-movement times and test Monte Carlo convergence.

    Each realization samples a warning-transmission time and response delay
    for every household, then expands the household values to all residents.
    Consecutive batches must satisfy the statistic and cumulative-curve
    tolerances for ``MC_STABLE_BATCHES_REQUIRED`` updates.
    """
    household_sizes = np.asarray(household_sizes, dtype=int)
    household_count = household_sizes.size
    resident_count = household_sizes.sum()
    # Map each resident to a household so members of the same household share
    # the same sampled value.
    household_of_resident = np.repeat(np.arange(household_count), household_sizes)
    # Zero-based rank at which 90% of residents have completed the process.
    t90_index = int(np.ceil(0.90 * resident_count)) - 1
    
    rng = np.random.default_rng(MC_RANDOM_SEED)

    # Retain the household-level realizations for representative selection.
    all_transmission_batches = []
    all_response_batches = []
    # Accumulate resident-level means, T90 values, and within-realization
    # maxima. Averaging the latter makes it a convergent expected statistic
    # rather than a global extreme that grows with the sample count.
    statistic_sums = np.zeros(6, dtype=float)
    # Count completions in 10-minute bins; the final bin collects later events.
    transmission_curve_counts = np.zeros(
        MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES + 1,
        dtype=np.int64,
    )
    response_curve_counts = np.zeros_like(transmission_curve_counts)
    samples = 0
    stable_batches = 0
    previous_summary = None
    previous_transmission_curve = None
    previous_response_curve = None
    transmission_curve_difference = np.inf
    response_curve_difference = np.inf
    converged = False

    while samples < MC_MAX_SAMPLES:
        batch_size = min(MC_BATCH_SIZE, MC_MAX_SAMPLES - samples)
        # Sample one Weibull transmission time and one Rayleigh response delay
        # per household and realization, in seconds.
        transmission_batch = (
            rng.weibull(alpha_time, size=(batch_size, household_count))
            * lam
            * 3600.0
        )
        response_batch = rng.rayleigh(
            scale=scale_response,
            size=(batch_size, household_count),
        ) * 60.0
        # Values on the same row belong to the same Monte Carlo realization.
        all_transmission_batches.append(transmission_batch)
        all_response_batches.append(response_batch)

        # Expand household values; residents are not sampled independently.
        transmission_residents = transmission_batch[:, household_of_resident]
        response_residents = response_batch[:, household_of_resident]

        statistic_sums += np.array(
            [
                transmission_residents.mean(axis=1).sum(),
                response_residents.mean(axis=1).sum(),
                np.partition(transmission_residents, t90_index, axis=1)[:, t90_index].sum(),
                np.partition(response_residents, t90_index, axis=1)[:, t90_index].sum(),
                transmission_residents.max(axis=1).sum(),
                response_residents.max(axis=1).sum(),
            ]
        )

        # Aggregate empirical CDFs in 10-minute bins without retaining every
        # realization-specific curve.
        transmission_time_bins = np.minimum(
            np.floor(
                transmission_residents / (60.0 * MC_CURVE_INTERVAL_MINUTES)
            ).astype(np.int64),
            MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES,
        )
        response_time_bins = np.minimum(
            np.floor(
                response_residents / (60.0 * MC_CURVE_INTERVAL_MINUTES)
            ).astype(np.int64),
            MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES,
        )
        transmission_curve_counts += np.bincount(
            transmission_time_bins.ravel(),
            minlength=MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES + 1,
        )
        response_curve_counts += np.bincount(
            response_time_bins.ravel(),
            minlength=MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES + 1,
        )
        samples += batch_size

        if samples < MC_MIN_SAMPLES:
            continue

        # Compare the current running summaries with the previous batch.
        summary = statistic_sums / samples
        transmission_curve = np.cumsum(transmission_curve_counts) / (
            samples * resident_count
        )
        response_curve = np.cumsum(response_curve_counts) / (
            samples * resident_count
        )
        # The first eligible batch establishes the comparison baseline.
        if previous_summary is not None:
            relative_change = np.max(
                np.abs(summary - previous_summary)
                / np.maximum(np.abs(previous_summary), 1.0)
            )
            transmission_curve_difference = np.max(
                np.abs(transmission_curve - previous_transmission_curve)
            )
            response_curve_difference = np.max(
                np.abs(response_curve - previous_response_curve)
            )
            # All statistic and cumulative-curve criteria must be satisfied.
            if (
                relative_change <= MC_RELATIVE_CHANGE_TOLERANCE
                and transmission_curve_difference <= 0.01
                and response_curve_difference <= 0.01
            ):
                stable_batches += 1
            else:
                stable_batches = 0
        previous_summary = summary
        previous_transmission_curve = transmission_curve
        previous_response_curve = response_curve

        # Stop early after the required number of consecutive stable batches.
        if stable_batches >= MC_STABLE_BATCHES_REQUIRED:
            converged = True
            break

    # Assemble all household-level realizations for representative selection.
    all_transmission_samples = np.vstack(all_transmission_batches)
    all_response_samples = np.vstack(all_response_batches)

    # Select the joint realization that minimizes the larger discrepancy from
    # the two mean cumulative curves, retaining paired transmission/response
    # values from the same realization.
    curve_bin_count = MC_CURVE_TIME_MAX_MINUTES // MC_CURVE_INTERVAL_MINUTES + 1

    def sample_curve_from_household_values(household_values):
        resident_values = household_values[household_of_resident]
        resident_bins = np.minimum(
            np.floor(
                resident_values / (60.0 * MC_CURVE_INTERVAL_MINUTES)
            ).astype(np.int64),
            curve_bin_count - 1,
        )
        return np.cumsum(np.bincount(resident_bins, minlength=curve_bin_count)) / resident_count

    sample_transmission_curves = np.zeros((samples, curve_bin_count), dtype=float)
    sample_response_curves = np.zeros_like(sample_transmission_curves)
    for sample_index in range(samples):
        sample_transmission_curves[sample_index] = sample_curve_from_household_values(
            all_transmission_samples[sample_index]
        )
        sample_response_curves[sample_index] = sample_curve_from_household_values(
            all_response_samples[sample_index]
        )

    def select_independent_curve_representatives(
        target_transmission_curve, target_response_curve
    ):
        transmission_differences = np.mean(
            np.abs(sample_transmission_curves - target_transmission_curve), axis=1
        )
        response_differences = np.mean(
            np.abs(sample_response_curves - target_response_curve), axis=1
        )
        transmission_index = int(np.argmin(transmission_differences))
        response_index = int(np.argmin(response_differences))
        score = (
            transmission_differences[transmission_index]
            + response_differences[response_index]
        )
        return (
            transmission_index,
            response_index,
            score,
            transmission_differences[transmission_index],
            response_differences[response_index],
        )

    representative_transmission_index, representative_response_index, representative_score, representative_transmission_difference, representative_response_difference = (
        select_independent_curve_representatives(transmission_curve, response_curve)
    )

    # q10/q90 use independent curve representatives for transmission and response.
    # For cumulative curves, faster scenarios have higher cumulative proportions at each time.
    # Therefore q10 (fast) targets the 90th percentile curve, and q90 (slow) targets the 10th percentile curve.
    curve_lower_quantile = 0.10
    curve_upper_quantile = 0.90
    transmission_fast_curve = np.quantile(
        sample_transmission_curves, curve_upper_quantile, axis=0
    )
    response_fast_curve = np.quantile(
        sample_response_curves, curve_upper_quantile, axis=0
    )
    transmission_slow_curve = np.quantile(
        sample_transmission_curves, curve_lower_quantile, axis=0
    )
    response_slow_curve = np.quantile(
        sample_response_curves, curve_lower_quantile, axis=0
    )

    q10_transmission_index, q10_response_index, q10_curve_score, q10_transmission_difference, q10_response_difference = (
        select_independent_curve_representatives(transmission_fast_curve, response_fast_curve)
    )
    q90_transmission_index, q90_response_index, q90_curve_score, q90_transmission_difference, q90_response_difference = (
        select_independent_curve_representatives(transmission_slow_curve, response_slow_curve)
    )
    # Keep T90 diagnostics for checking the selected curve-based scenarios.
    transmission_t90_samples = np.partition(
        all_transmission_samples[:, household_of_resident], t90_index, axis=1
    )[:, t90_index]
    response_t90_samples = np.partition(
        all_response_samples[:, household_of_resident], t90_index, axis=1
    )[:, t90_index]

    def expand_household_sample(household_values):
        """Expand one household-level realization to all residents."""
        return household_values[household_of_resident].copy()

    # ``mean`` is closest to the Monte Carlo mean cumulative curves. The q10
    # and q90 labels represent relatively fast and slow scenarios selected
    # against the 85th- and 15th-percentile cumulative curves, respectively.
    # Transmission and response are selected independently and may therefore
    # originate from different Monte Carlo realization indices.
    scenario_times = {
        "mean": (
            expand_household_sample(all_transmission_samples[representative_transmission_index]),
            expand_household_sample(all_response_samples[representative_response_index]),
        ),
        "q10": (
            expand_household_sample(all_transmission_samples[q10_transmission_index]),
            expand_household_sample(all_response_samples[q10_response_index]),
        ),
        "q90": (
            expand_household_sample(all_transmission_samples[q90_transmission_index]),
            expand_household_sample(all_response_samples[q90_response_index]),
        ),
    }

    # Diagnostics are written to timestatistics.mat for convergence and scenario checks.
    # mc_response_* corresponds to response delay only, not transmission + response.
    diagnostics = {
        "mc_samples": samples,
        "mc_converged": int(converged),
        "mc_transmission_mean": summary[0],
        "mc_response_mean": summary[1],
        "mc_transmission_t90_mean": summary[2],
        "mc_response_t90_mean": summary[3],
        "mc_transmission_max_mean": summary[4],
        "mc_response_max_mean": summary[5],
        "mc_transmission_curve_max_change": transmission_curve_difference,
        "mc_response_curve_max_change": response_curve_difference,
        "mc_representative_transmission_sample_index": representative_transmission_index,
        "mc_representative_response_sample_index": representative_response_index,
        "mc_representative_sample_index": representative_transmission_index,
        "mc_representative_curve_score": representative_score,
        "mc_representative_transmission_curve_difference": representative_transmission_difference,
        "mc_representative_response_curve_difference": representative_response_difference,
        "mc_q10_transmission_curve_sample_index": q10_transmission_index,
        "mc_q10_response_curve_sample_index": q10_response_index,
        "mc_q90_transmission_curve_sample_index": q90_transmission_index,
        "mc_q90_response_curve_sample_index": q90_response_index,
        "mc_q10_curve_sample_index": q10_transmission_index,
        "mc_q90_curve_sample_index": q90_transmission_index,
        "mc_q10_curve_score": q10_curve_score,
        "mc_q90_curve_score": q90_curve_score,
        "mc_q10_transmission_curve_difference": q10_transmission_difference,
        "mc_q10_response_curve_difference": q10_response_difference,
        "mc_q90_transmission_curve_difference": q90_transmission_difference,
        "mc_q90_response_curve_difference": q90_response_difference,
        "mc_q10_curve_target_quantile": curve_upper_quantile,
        "mc_q90_curve_target_quantile": curve_lower_quantile,
        # Compatibility aliases: singular sample_index aliases refer to transmission.
        "mc_transmission_q10_sample_index": q10_transmission_index,
        "mc_transmission_q90_sample_index": q90_transmission_index,
        "mc_response_q10_sample_index": q10_response_index,
        "mc_response_q90_sample_index": q90_response_index,
        "mc_transmission_t90_q10": transmission_t90_samples[q10_transmission_index],
        "mc_transmission_t90_q90": transmission_t90_samples[q90_transmission_index],
        "mc_response_t90_q10": response_t90_samples[q10_response_index],
        "mc_response_t90_q90": response_t90_samples[q90_response_index],
    }
    return scenario_times, diagnostics


def sample_converged_representative_speeds(
    number_of_kids,
    number_of_adults,
    alpha_kids,
    beta_kids,
    alpha_adults,
    beta_adults,
    output_directory,
):
    """Select a complete representative speed sample for each group."""

    def closest_sample(sample_batches, target_curve):
        best_sample = None
        best_index = -1
        best_score = np.inf
        sample_offset = 0
        for sample_batch in sample_batches:
            ordered_batch = np.sort(sample_batch, axis=1)
            scores = np.mean(np.abs(ordered_batch - target_curve), axis=1)
            local_index = int(np.argmin(scores))
            local_score = float(scores[local_index])
            if local_score < best_score:
                best_sample = sample_batch[local_index].copy()
                best_index = sample_offset + local_index
                best_score = local_score
            sample_offset += sample_batch.shape[0]
        return best_sample, best_index, best_score

    number_of_kids = int(number_of_kids)
    number_of_adults = int(number_of_adults)
    if number_of_kids <= 0 or number_of_adults <= 0:
        raise ValueError("Both population groups must contain at least one person.")
    if min(alpha_kids, beta_kids, alpha_adults, beta_adults) <= 0.0:
        raise ValueError("All Weibull distribution parameters must be positive.")

    kids_t90_index = int(np.ceil(0.90 * number_of_kids)) - 1
    adults_t90_index = int(np.ceil(0.90 * number_of_adults)) - 1
    rng = np.random.default_rng(SPEED_MC_RANDOM_SEED)
    kids_batches = []
    adults_batches = []
    kids_ordered_sum = np.zeros(number_of_kids, dtype=np.float64)
    adults_ordered_sum = np.zeros(number_of_adults, dtype=np.float64)
    statistic_sums = np.zeros(6, dtype=np.float64)

    samples = 0
    stable_batches = 0
    previous_summary = None
    previous_kids_curve = None
    previous_adults_curve = None
    relative_change = np.inf
    kids_curve_difference = np.inf
    adults_curve_difference = np.inf
    converged = False

    while samples < MC_MAX_SAMPLES:
        batch_size = min(MC_BATCH_SIZE, MC_MAX_SAMPLES - samples)
        kids_batch = (
            rng.weibull(alpha_kids, (batch_size, number_of_kids))
            * beta_kids
        )
        adults_batch = (
            rng.weibull(alpha_adults, (batch_size, number_of_adults))
            * beta_adults
        )
        kids_batches.append(kids_batch)
        adults_batches.append(adults_batch)

        ordered_kids = np.sort(kids_batch, axis=1)
        ordered_adults = np.sort(adults_batch, axis=1)
        kids_ordered_sum += ordered_kids.sum(axis=0)
        adults_ordered_sum += ordered_adults.sum(axis=0)
        statistic_sums += np.array(
            [
                kids_batch.mean(axis=1).sum(),
                adults_batch.mean(axis=1).sum(),
                ordered_kids[:, kids_t90_index].sum(),
                ordered_adults[:, adults_t90_index].sum(),
                ordered_kids[:, -1].sum(),
                ordered_adults[:, -1].sum(),
            ]
        )
        samples += batch_size
        if samples < MC_MIN_SAMPLES:
            continue

        summary = statistic_sums / float(samples)
        kids_curve = kids_ordered_sum / float(samples)
        adults_curve = adults_ordered_sum / float(samples)
        if previous_summary is not None:
            relative_change = float(
                np.max(
                    np.abs(summary - previous_summary)
                    / np.maximum(np.abs(previous_summary), 0.1)
                )
            )
            kids_curve_difference = float(
                np.max(np.abs(kids_curve - previous_kids_curve))
            )
            adults_curve_difference = float(
                np.max(np.abs(adults_curve - previous_adults_curve))
            )
            if (
                relative_change <= MC_RELATIVE_CHANGE_TOLERANCE
                and kids_curve_difference
                <= SPEED_MC_CURVE_DIFFERENCE_TOLERANCE
                and adults_curve_difference
                <= SPEED_MC_CURVE_DIFFERENCE_TOLERANCE
            ):
                stable_batches += 1
            else:
                stable_batches = 0

        previous_summary = summary.copy()
        previous_kids_curve = kids_curve.copy()
        previous_adults_curve = adults_curve.copy()
        if stable_batches >= MC_STABLE_BATCHES_REQUIRED:
            converged = True
            break

    if previous_summary is None:
        raise RuntimeError("Insufficient speed samples for convergence assessment.")

    speed_kids, kids_index, kids_score = closest_sample(
        kids_batches, previous_kids_curve
    )
    speed_adults, adults_index, adults_score = closest_sample(
        adults_batches, previous_adults_curve
    )
    diagnostics = {
        "speed_mc_samples": int(samples),
        "speed_mc_converged": int(converged),
        "speed_mc_relative_statistic_change": float(relative_change),
        "speed_mc_kids_curve_max_change_mps": float(kids_curve_difference),
        "speed_mc_adults_curve_max_change_mps": float(
            adults_curve_difference
        ),
        "speed_mc_kids_mean_mps": float(previous_summary[0]),
        "speed_mc_adults_mean_mps": float(previous_summary[1]),
        "speed_mc_kids_p90_mean_mps": float(previous_summary[2]),
        "speed_mc_adults_p90_mean_mps": float(previous_summary[3]),
        "speed_mc_kids_max_mean_mps": float(previous_summary[4]),
        "speed_mc_adults_max_mean_mps": float(previous_summary[5]),
        "speed_mc_kids_selected_sample_index": int(kids_index),
        "speed_mc_adults_selected_sample_index": int(adults_index),
        "speed_mc_kids_selection_mae_mps": float(kids_score),
        "speed_mc_adults_selection_mae_mps": float(adults_score),
        "speed_mc_kids_selected_mean_mps": float(np.mean(speed_kids)),
        "speed_mc_adults_selected_mean_mps": float(np.mean(speed_adults)),
    }

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_directory / "representative_speed_sample.npz",
        speed_kids_mps=speed_kids,
        speed_adults_mps=speed_adults,
        kids_mean_ordered_speed_curve_mps=previous_kids_curve,
        adults_mean_ordered_speed_curve_mps=previous_adults_curve,
    )
    metadata = {
        "selection_principle": (
            "The two population groups are selected separately as the real "
            "samples closest to their converged mean ordered-speed curves."
        ),
        "random_seed": SPEED_MC_RANDOM_SEED,
        "number_of_kids": number_of_kids,
        "number_of_adults": number_of_adults,
        "kids_weibull_shape": float(alpha_kids),
        "kids_weibull_scale_mps": float(beta_kids),
        "adults_weibull_shape": float(alpha_adults),
        "adults_weibull_scale_mps": float(beta_adults),
        "convergence": diagnostics,
    }
    with (
        output_directory / "representative_speed_sample.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    return speed_kids, speed_adults, diagnostics


# =============================================================================
# Performance-optimization functions
# =============================================================================
# The base model ultimately discards pedestrian interactions beyond 5 m, but
# first evaluates every pedestrian pair. It also constructs a dense matrix for
# all active pedestrians and obstacle points. The optimized functions below use
# Numba and cKDTree to evaluate only interactions that can contribute. Runtime
# monkey patches affect this process only and do not modify pysocialforce files.


@njit(cache=False)
def _social_force_kernel(
    pos,
    vel,
    source_enabled,
    lambda_importance,
    gamma,
    n,
    n_prime,
):
    """Evaluate the original social-force equations within the 5 m cutoff."""
    pedestrian_count = pos.shape[0]
    output = np.zeros((pedestrian_count, 2), dtype=np.float64)

    for i in range(pedestrian_count):
        # Skip force rows for pedestrians already within 3 m of the shelter;
        # they remain present as neighbours in other pedestrians' interactions.
        if not source_enabled[i]:
            continue

        for j in range(pedestrian_count):
            if i == j:
                continue

            diff_x = pos[i, 0] - pos[j, 0]
            diff_y = pos[i, 1] - pos[j, 1]
            diff_length = np.sqrt(diff_x * diff_x + diff_y * diff_y)

            # The base implementation sets interactions beyond 5 m to zero.
            if diff_length > 5.0:
                continue

            if diff_length > 0.0:
                direction_x = diff_x / diff_length
                direction_y = diff_y / diff_length
            else:
                direction_x = 0.0
                direction_y = 0.0

            velocity_diff_x = vel[j, 0] - vel[i, 0]
            velocity_diff_y = vel[j, 1] - vel[i, 1]
            interaction_x = (
                lambda_importance * velocity_diff_x + direction_x
            )
            interaction_y = (
                lambda_importance * velocity_diff_y + direction_y
            )
            interaction_length = np.sqrt(
                interaction_x * interaction_x
                + interaction_y * interaction_y
            )

            # Match the base normalization behavior for a zero vector.
            if interaction_length == 0.0:
                continue

            interaction_direction_x = interaction_x / interaction_length
            interaction_direction_y = interaction_y / interaction_length
            theta = (
                np.arctan2(
                    interaction_direction_y,
                    interaction_direction_x,
                )
                - np.arctan2(direction_y, direction_x)
            )
            B = gamma * interaction_length
            velocity_amount = np.exp(
                -diff_length / B
                - (n_prime * B * theta) ** 2
            )
            angle_amount = -np.sign(theta) * np.exp(
                -diff_length / B
                - (n * B * theta) ** 2
            )

            output[i, 0] += (
                velocity_amount * interaction_direction_x
                - angle_amount * interaction_direction_y
            )
            output[i, 1] += (
                velocity_amount * interaction_direction_y
                + angle_amount * interaction_direction_x
            )

    return output


def _fast_social_force(self, stepn):
    """Compute social forces within 5 m using the compiled kernel."""
    lambda_importance = self.config("lambda_importance", 2.0)
    gamma = self.config("gamma", 0.35)
    n = self.config("n", 2)
    n_prime = self.config("n_prime", 3)

    elapsed_time = (stepn + 1) * self.peds.step_width
    active_mask = self.peds.pretime <= elapsed_time
    active_indices = np.flatnonzero(active_mask)
    force = np.zeros((self.peds.size(), 2), dtype=float)

    if active_indices.size == 0:
        return force

    pos = self.peds.pos()[active_mask]
    vel = self.peds.vel()[active_mask]

    # Cache shelter distances so both the force kernel and pedestrian update
    # use the same nearest-neighbour query for this step.
    all_distance_to_shelter, _ = self._fast_goal_tree.query(
        self.peds.pos(), k=1
    )
    self.peds._fast_goal_distance = all_distance_to_shelter
    self.peds._fast_goal_distance_step = stepn
    source_enabled = all_distance_to_shelter[active_mask] >= 3.0

    # The compiled loop applies the 5 m cutoff before evaluating the equations,
    # avoiding repeated dense N-by-(N-1) temporary arrays.
    active_force = _social_force_kernel(
        pos,
        vel,
        source_enabled,
        lambda_importance,
        gamma,
        n,
        n_prime,
    )

    force[active_indices] = active_force
    return force * 4.5


def _fast_obstacle_force(self, stepn):
    """Evaluate obstacle forces only for points within the search radius."""
    sigma = 0.1
    elapsed_time = (stepn + 1) * self.peds.step_width
    active_mask = self.peds.pretime <= elapsed_time
    active_indices = np.flatnonzero(active_mask)
    force = np.zeros((self.peds.size(), 2), dtype=float)

    if active_indices.size == 0 or self._fast_obstacles.shape[0] == 0:
        return force

    pos = self.peds.pos()[active_mask]

    # Query nearby obstacles before evaluating directions and exponential terms.
    neighbour_lists = self._fast_obstacle_tree.query_ball_point(
        pos, r=self._fast_obstacle_search_radius
    )
    counts = np.fromiter(
        (len(item) for item in neighbour_lists),
        dtype=np.intp,
        count=len(neighbour_lists),
    )
    if counts.sum() == 0:
        return force

    local_ped_indices = np.repeat(np.arange(active_indices.size), counts)
    obstacle_indices = np.concatenate(
        [np.asarray(item, dtype=np.intp) for item in neighbour_lists if item]
    )
    diff = (
        self._fast_obstacles[obstacle_indices]
        - pos[local_ped_indices]
    )
    distance = np.linalg.norm(diff, axis=1)
    direction = np.zeros_like(diff)
    nonzero_distance = distance > 0
    direction[nonzero_distance] = (
        diff[nonzero_distance] / distance[nonzero_distance, np.newaxis]
    )
    adjusted_distance = distance - self.peds.agent_radius
    contribution = direction * np.exp(
        -adjusted_distance[:, np.newaxis] / sigma
    )

    local_force = np.zeros((active_indices.size, 2), dtype=float)
    local_force[:, 0] = np.bincount(
        local_ped_indices,
        weights=contribution[:, 0],
        minlength=active_indices.size,
    )
    local_force[:, 1] = np.bincount(
        local_ped_indices,
        weights=contribution[:, 1],
        minlength=active_indices.size,
    )

    # Preserve the index-assignment behavior of the base forces.py implementation.
    force_ped = np.zeros((self.peds.size(), 2), dtype=float)
    force_ped[:active_indices.size] = local_force
    force[active_indices] = force_ped[active_indices]
    return force * 3.0


def _fast_ped_step(self, force, stepn, groups=None):
    """Optimize road correction and shelter-arrival checks."""
    elapsed_time = (stepn + 1) * self.step_width
    active_mask = self.pretime <= elapsed_time
    active_indices = np.flatnonzero(active_mask)

    desired_velocity = self.vel() + self.step_width * force
    desired_velocity = self.capped_velocity(desired_velocity, self.max_speeds)
    desired_velocity[
        stateutils.desired_directions(self.state)[1] < 0.5
    ] = [0, 0]

    # Reuse shelter distances computed during the force calculation.
    if getattr(self, "_fast_goal_distance_step", None) == stepn:
        distance_to_shelter = self._fast_goal_distance
    else:
        distance_to_shelter, _ = self._fast_goal_tree.query(
            self.state[:, 0:2], k=1
        )
    indices_stop = distance_to_shelter < 3.0
    next_stop_id = self.stopId
    next_stop_id[indices_stop] += 1

    target_last = self.state[:, 7:8]
    next_state = self.state

    # Preserve the original active-pedestrian branch behavior.
    if active_indices.any():
        new_pos = (
            next_state[active_indices, 0:2]
            + desired_velocity[active_indices] * self.step_width
        )

        col_id = np.minimum(
            ((new_pos[:, 0] - self.minXroad) // self.pixelWidthroad).astype(int),
            self.roadPolygon.shape[1] - 1,
        )
        row_id = np.minimum(
            ((self.maxYroad - new_pos[:, 1]) // -self.pixelHeightroad).astype(int),
            self.roadPolygon.shape[0] - 1,
        )
        road_id = self.roadPolygon[row_id, col_id]
        outside_local_indices = np.flatnonzero(road_id == 0)

        # Query the nearest road center only for pedestrians outside the road.
        if outside_local_indices.size:
            _, nearest_center = self._fast_roadcenter_tree.query(
                new_pos[outside_local_indices], k=1
            )
            new_pos[outside_local_indices] = self.roadcenterPos[nearest_center]

        next_state[active_indices, 0:2] = new_pos
        next_state[active_indices, 2:4] = desired_velocity[active_indices]
        next_target = stateutils.target(
            next_state[:, 0:2],
            target_last,
            self.coorpath,
            self.guidelineIndex,
        )[0]
        next_state[active_indices, 7:8] = next_target[
            active_indices
        ].reshape(-1, 1)

    next_groups = self.groups if groups is None else groups
    self.update(next_state, next_groups, next_stop_id)


def enable_fast_mode(simulator):
    """Install optimized runtime kernels on one Simulator instance."""
    peds = simulator.peds
    simulator._fast_original_forces = list(simulator.forces)

    # Compile the Numba kernel before timing the simulation.
    _social_force_kernel(
        np.zeros((1, 2), dtype=float),
        np.zeros((1, 2), dtype=float),
        np.ones(1, dtype=np.bool_),
        2.0,
        0.35,
        2,
        3,
    )

    # Build the static spatial indices once and reuse them for all time steps.
    peds._fast_goal_tree = cKDTree(np.asarray(peds.goal()))
    peds._fast_roadcenter_tree = cKDTree(np.asarray(peds.roadcenterPos))
    peds.step = MethodType(_fast_ped_step, peds)

    optimized_forces = []
    group_force_types = (
        force_module.GroupCoherenceForceAlt,
        force_module.GroupRepulsiveForce,
        force_module.GroupGazeForceAlt,
    )

    for force_item in simulator.forces:
        if isinstance(force_item, force_module.SocialForce):
            force_item._fast_goal_tree = peds._fast_goal_tree
            force_item._get_force = MethodType(
                _fast_social_force, force_item
            )
        elif isinstance(force_item, force_module.ObstacleForce):
            obstacles = np.vstack(simulator.get_obstacles())
            force_item._fast_obstacles = np.asarray(obstacles)
            force_item._fast_obstacle_tree = cKDTree(
                force_item._fast_obstacles
            )

            # Equivalent search radius for the base obstacle-distance condition:
            # (obstacle distance - agent_radius) < threshold - 3.05.
            threshold = (
                force_item.config("threshold", 0.2)
                + peds.agent_radius
            )
            force_item._fast_obstacle_search_radius = max(
                0.0,
                threshold - 3.05 + peds.agent_radius,
            )
            force_item._get_force = MethodType(
                _fast_obstacle_force, force_item
            )

        # Group forces are identically zero when no pedestrian groups are defined.
        if not peds.groups and isinstance(force_item, group_force_types):
            continue
        optimized_forces.append(force_item)

    simulator.forces = optimized_forces
    return simulator


def disable_fast_mode(simulator):
    """Remove runtime patches before serializing the Simulator."""
    peds = simulator.peds

    # Removing the instance method restores the class-level PedState.step method.
    if "step" in peds.__dict__:
        delattr(peds, "step")

    for attr_name in (
        "_fast_goal_tree",
        "_fast_roadcenter_tree",
        "_fast_goal_distance",
        "_fast_goal_distance_step",
    ):
        if hasattr(peds, attr_name):
            delattr(peds, attr_name)

    original_forces = getattr(
        simulator, "_fast_original_forces", simulator.forces
    )
    for force_item in original_forces:
        if "_get_force" in force_item.__dict__:
            delattr(force_item, "_get_force")
        for attr_name in (
            "_fast_goal_tree",
            "_fast_obstacles",
            "_fast_obstacle_tree",
            "_fast_obstacle_search_radius",
        ):
            if hasattr(force_item, attr_name):
                delattr(force_item, attr_name)

    simulator.forces = original_forces
    if hasattr(simulator, "_fast_original_forces"):
        delattr(simulator, "_fast_original_forces")
    return simulator


if __name__ == "__main__":

    fixed_python_rng = random.Random(FIXED_RANDOM_SEED)
    fixed_numpy_rng = np.random.RandomState(FIXED_RANDOM_SEED)

    # Create output directories when each scenario starts.
     
    num_evacuees = 198
    # Weibull parameters for warning-transmission time.
    alpha_time = 0.6
    beta_time = 3.5
    lam = beta_time ** (-1 / alpha_time)
    
    scale_response = 18.2
    
    # Read household departure locations and construct initial positions and
    # route identifiers for 198 residents.
    filedeparture = DATA_DIR / "Departure" / "Departure.shp"
    departureattribute = gpd.read_file(filedeparture)
    Id = departureattribute['Id'].values
    
    startpositinX = np.zeros(num_evacuees)
    startpositinY = np.zeros(num_evacuees)
    
    guidelineIndex = np.zeros(num_evacuees)
    household_sizes = []
    for i in range(len(departureattribute)):
    
        maxX = departureattribute.bounds.iloc[i]['maxx']
        minX = departureattribute.bounds.iloc[i]['minx']

        maxY = departureattribute.bounds.iloc[i]['maxy']
        minY = departureattribute.bounds.iloc[i]['miny']

        Xseris = np.linspace(minX, maxX, 3)
        Yseris = np.linspace(maxY, minY, 3)
        
        XX, YY = np.meshgrid(Xseris, Yseris)
        
        if i <= 11:
            household_size = 5
            x = XX.flatten()[:5]
            y = YY.flatten()[:5]
            startpositinX[i*5:i*5+5] = x
            startpositinY[i*5:i*5+5] = y
            
            if Id[i] <= 6:
                guidelineIndex[i*5:i*5+5] = 1
            elif Id[i] >= 7 and Id[i] <= 12:
                guidelineIndex[i*5:i*5+5] = 2
            elif Id[i] >= 13 and Id[i] <= 16:
                guidelineIndex[i*5:i*5+5] = 3
            elif Id[i] >= 17 and Id[i] <= 20:
                guidelineIndex[i*5:i*5+5] = 4
            elif Id[i] >= 21 and Id[i] <= 23:
                guidelineIndex[i*5:i*5+5] = 5
            else:
                guidelineIndex[i*5:i*5+5] = 6
        else:
            household_size = 6
            x = XX.flatten()[:6]
            y = YY.flatten()[:6]
            startpositinX[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = x
            startpositinY[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = y
            
            if Id[i] <= 6:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 1
            elif Id[i] >= 7 and Id[i] <= 12:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 2
            elif Id[i] >= 13 and Id[i] <= 16:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 3
            elif Id[i] >= 17 and Id[i] <= 20:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 4
            elif Id[i] >= 21 and Id[i] <= 23:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 5
            else:
                guidelineIndex[11*5+5 + (i-12)*6 :11*5+5 + (i-12)*6+6] = 6

        household_sizes.append(household_size)

    if sum(household_sizes) != num_evacuees:
        raise ValueError("The household sizes do not sum to num_evacuees.")

    # Sample pre-movement times and walking speeds with independent Monte Carlo
    # streams; shelter locations and final speed shuffling use fixed seeds.
        
    scenario_times, mc_diagnostics = (
        sample_converged_premovement_times(
            household_sizes,
            alpha_time,
            lam,
            scale_response,
        )
    )
    
    guidelineIndex = guidelineIndex.astype(int)
    print(
        "MC convergence: samples={mc_samples}, converged={mc_converged}, "
        "transmission_mean={mc_transmission_mean:.2f}s, "
        "response_mean={mc_response_mean:.2f}s".format(**mc_diagnostics)
    )

    # Select a target shelter cell for each resident using the fixed seed.
    shelterAssemble = []
    with rasterio.open(DATA_DIR / "Shelter" / "Shelter.tif") as src:
        shelterZone = src.read(1)
        
        transform = src.transform
        minX = transform[2]
        maxY = transform[5]
        pixelWidth = transform[0]
        pixelHeight = transform[4]
        ncol = src.width
        nrow = src.height
        maxX = minX + (ncol-1)*pixelWidth
        minY = maxY + (nrow-1)*pixelHeight
        
        Xseris = np.linspace(minX, maxX, ncol)
        Yseris = np.linspace(maxY, minY, nrow)
        
        XX, YY = np.meshgrid(Xseris, Yseris)
        
        shelterZone[np.where(shelterZone == 1)] = 2
        shelterZone[np.where(shelterZone == 0)] = 1
        shelterZone[np.where(shelterZone == 2)] = 0

        shelterMask = np.where(shelterZone == 1, True, False)  
        
        XX_shelter = XX[shelterMask]
        YY_shelter = YY[shelterMask]
        
        num = XX_shelter.shape[0]
        randInd = [
            fixed_python_rng.randint(0, num - 1)
            for _ in range(num_evacuees)
        ]
    print(randInd[0])
    
    arr5 = XX_shelter[randInd]
    arr6 = YY_shelter[randInd]
    shelterPos = np.stack((arr5, arr6), axis=1)
    shelterAssemble.append(shelterPos)
    shelterAssemble = np.array(shelterAssemble)

    # Walking-speed distribution parameters (Rinne et al., 2010). Sample the
    # two groups separately and select the complete realization closest to each
    # converged mean ordered-speed curve.
    number_of_kids = num_evacuees // 3
    number_of_adults = num_evacuees - number_of_kids
    alpha_kids = 10.14  # Weibull shape parameter
    beta_kids = 1.41  # Weibull scale parameter
    alpha_adults = 7.62  # Weibull shape parameter
    beta_adults = 1.60  # Weibull scale parameter
    speed_kids, speed_adults, speed_mc_diagnostics = (
        sample_converged_representative_speeds(
            number_of_kids,
            number_of_adults,
            alpha_kids,
            beta_kids,
            alpha_adults,
            beta_adults,
            RESULTS_DIR,
        )
    )

    arrv = np.concatenate((speed_kids, speed_adults))
    fixed_numpy_rng.shuffle(arrv)
    print(arrv[0])
    direction, dist = stateutils.normalize(np.stack((arr5, arr6), axis=1) - np.stack((startpositinX, startpositinY), axis=1))              
    arr3 = direction[:,0]*arrv  
    arr4 = direction[:,1]*arrv

    initial_state = np.stack((startpositinX, startpositinY, arr3, arr4, shelterAssemble[0,:,0], shelterAssemble[0,:,1]), axis=1)
    groups = []
    
    # Read the coordinate sequences of the six evacuation guide paths.
    coorpath = [dict(x=None) for _ in range(6)]
    for i in range(6):
        filepath = DATA_DIR / "GuideLine" / f"GuidePoint{i + 1}.shp"
        Attribute = gpd.read_file(filepath)
        x_coords = Attribute.geometry.x
        y_coords = Attribute.geometry.y
        x_coords = x_coords.values
        y_coords = y_coords.values
        coorpathi = np.stack((x_coords, y_coords), axis=1)
        coorpath[i]['coor'] = coorpathi
        
    # Find the nearest initial guide point for each resident.
    initialguidepointIndex = np.zeros((num_evacuees,1))
    for i in range(num_evacuees):
        lineId = guidelineIndex[i]
        positioni = np.stack((startpositinX[i], startpositinY[i])).reshape(1, 2)
        coorpathi = coorpath[lineId-1]['coor']
        distance = np.linalg.norm(positioni - coorpathi, axis = 1)
        initialguidepointIndex[i] = np.argmin(distance)
    initialguidepointIndex = initialguidepointIndex.astype(int)
    
    # Extract obstacle coordinates from the road-boundary raster.
    with rasterio.open(DATA_DIR / "Road" / "BoundaryObstacle.tif") as src:
        road = src.read(1)
        
        transform = src.transform
        minX = transform[2]
        maxY = transform[5]
        pixelWidth = transform[0]
        pixelHeight = transform[4]
        ncol = src.width
        nrow = src.height
        maxX = minX + (ncol-1)*pixelWidth
        minY = maxY + (nrow-1)*pixelHeight
        
        Xseris = np.linspace(minX, maxX, ncol)
        Yseris = np.linspace(maxY, minY, nrow)
        
        XX, YY = np.meshgrid(Xseris, Yseris)
            
        road[np.where(road != road[0,0])] = 1
        road[np.where(road == road[0,0])] = 0  
        roadBoundMask = np.where(road > 0, True, False)  
        
        XX_road = XX[roadBoundMask]
        YY_road = YY[roadBoundMask]
        
    obs = np.array(
                list(
                    zip(XX_road, YY_road)
                )
            )
        
    stopId = np.zeros(num_evacuees, dtype = int)
    
    with rasterio.open(DATA_DIR / "Road" / "RoadPolygon.tif") as src:
        roadPolygon = src.read(1)
        
        roadPolygon[np.where(roadPolygon == 1)] = 2
        roadPolygon[np.where(roadPolygon == 0)] = 1
        roadPolygon[np.where(roadPolygon == 2)] = 0
        
        transform = src.transform
        minXroad = transform[2]
        maxYroad = transform[5]
        pixelWidthroad = transform[0]
        pixelHeightroad = transform[4]
        ncolroad = src.width
        nrowroad = src.height
        maxXroad = minXroad + (ncolroad-1)*pixelWidthroad
        minYroad = maxYroad + (nrowroad-1)*pixelHeightroad
        
        Xserisroad = np.linspace(minXroad, maxXroad, ncolroad)
        Yserisroad = np.linspace(maxYroad, minYroad, nrowroad)
        
        XXroad, YYroad = np.meshgrid(Xserisroad, Yserisroad)
        
        XXroad = XXroad.flatten()
        YYroad = YYroad.flatten()
        roadPos = np.column_stack((XXroad, YYroad))

    # Read the road-center raster used to return off-road pedestrians to the
    # nearest centerline point.
    with rasterio.open(DATA_DIR / "Road" / "Road_Center.tif") as src:
        roadPolygonCenter = src.read(1)
        roadPolygonCenter[np.where(roadPolygonCenter < 15)] = 1
        roadPolygonCenter[np.where(roadPolygonCenter == 15)] = 0
        
        transformcenter = src.transform
        minXroadcenter = transformcenter[2]
        maxYroadcenter = transformcenter[5]
        pixelWidthroadcenter = transformcenter[0]
        pixelHeightroadcenter = transformcenter[4]
        ncolroadcenter = src.width
        nrowroadcenter = src.height
        maxXroadcenter = minXroadcenter + (ncolroadcenter-1)*pixelWidthroadcenter
        minYroadcenter = maxYroadcenter + (nrowroadcenter-1)*pixelHeightroadcenter
        
        Xserisroadcenter = np.linspace(minXroadcenter, maxXroadcenter, ncolroadcenter)
        Yserisroadcenter = np.linspace(maxYroadcenter, minYroadcenter, nrowroadcenter)
        
        XXroadcenter, YYroadcenter = np.meshgrid(Xserisroadcenter, Yserisroadcenter)
        
        roadMask = np.where(roadPolygonCenter == 1, True, False) 
        XXroadcenter = XXroadcenter[roadMask]
        YYroadcenter = YYroadcenter[roadMask]
        roadcenterPos = np.column_stack((XXroadcenter, YYroadcenter))
        
    # The three scenarios share spatial and speed inputs and differ only in
    # household-level warning-transmission and response times.
    for scenario_name, (transmission_time, response_time) in scenario_times.items():
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        pre_time = transmission_time + response_time

        start_time = time.process_time()
        s = psf.Simulator(
            initial_state.copy(),
            stopId.copy(),
            groups=groups,
            obstacles=obs,
            pretime=pre_time,
            coorpath=coorpath,
            initialguidepoint=initialguidepointIndex.copy(),
            guidelineIndex=guidelineIndex.copy(),
            roadPolygon=roadPolygon,
            roadPos=roadPos,
            roadcenterPos=roadcenterPos,
            minXroad=minXroad,
            maxYroad=maxYroad,
            pixelWidthroad=pixelWidthroad,
            pixelHeightroad=pixelHeightroad,
            config_file=Path(__file__).resolve().parent.joinpath("code.toml"),
        )

        # Apply the same optimized kernels and time each scenario independently.
        enable_fast_mode(s)
        s.step(SIMULATION_STEPS)
        
        simulation_time = time.process_time() - start_time
        print(f"simulation time: {simulation_time:.2f} s")
        disable_fast_mode(s)

        # Save time inputs, the Simulator object, state trajectories, and
        # optional animation separately for each scenario.
        scenario_diagnostics = {
            **mc_diagnostics,
            **speed_mc_diagnostics,
            "scenario_name": scenario_name,
        }
        time_statistics = {
            "simulation_time": simulation_time,
            "pre_time": pre_time,
            "transmission_time": transmission_time,
            "response_time": response_time,
            **scenario_diagnostics,
        }
        sio.savemat(
            RESULTS_DIR.joinpath(f"timestatistics_mc_wr_{scenario_name}.mat"),
            time_statistics,
        )
        with open(
            RESULTS_DIR.joinpath(f"simulator_mc_wr_{scenario_name}.pkl"), "wb"
        ) as file:
            pickle.dump(s, file)

        states, group_states = s.get_states()
        np.save(RESULTS_DIR.joinpath(f"states_mc_wr_{scenario_name}.npy"), states)
        
        stopId_states = s.peds.stopId_states
        stopId2matrix= {'stopId2matrix': stopId_states}
        sio.savemat(
            RESULTS_DIR.joinpath(f"stopId_mc_wr_{scenario_name}.mat"),
            stopId2matrix,
        )
    
        with psf.plot.SceneVisualizer(
            s, str(IMAGES_DIR.joinpath(f"animation_mc_wr_{scenario_name}"))
        ) as sv:
            sv.animate()
        plot_time = time.process_time() - start_time
        print(f"plot time: {plot_time:.2f} s")

    num_evacuees = 198

        
    
       
   
    
