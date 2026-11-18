"""
Autonomous Deep-Space Spacecraft Navigation Simulation
======================================================
Reproducible synthetic-observation study using nonlinear least squares.

Run from the repository root:
    python src/navigation_simulation.py

The script writes plots and CSV summaries to the repository's results/ folder.
This is a mathematical simulation, not a flight-validated navigation system.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

# ------------------------------ Configuration ------------------------------
RANDOM_SEED = 20261102
TRUE_POSITION = np.array([100.0, 50.0, 20.0])
INITIAL_GUESS = np.array([120.0, 30.0, 50.0])
TRIALS_PER_CONDITION = 100
OBSERVATION_COUNTS = [2, 3, 5, 10, 20]
UNCERTAINTY_LEVELS = [0.00, 0.01, 0.05, 0.10, 0.25]
EXPERIMENT_1_SIGMA = 0.05

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPOSITORY_ROOT / "results"


def create_celestial_objects():
    """Create the 20 synthetic celestial-object coordinates used in the study."""
    return {
        f"Object {i + 1}": np.array([
            300.0 * np.cos(2 * np.pi * i / 20),
            300.0 * np.sin(2 * np.pi * i / 20),
            100.0 * np.sin(4 * np.pi * i / 20),
        ])
        for i in range(20)
    }


CELESTIAL_OBJECTS = create_celestial_objects()


# --------------------------- Observation geometry --------------------------
def calculate_direction(object_position, spacecraft_position):
    """Return the unit line-of-sight vector from spacecraft to object."""
    relative = np.asarray(object_position) - np.asarray(spacecraft_position)
    distance = np.linalg.norm(relative)
    if not np.isfinite(distance) or distance == 0:
        raise ValueError("Spacecraft and object positions must be distinct.")
    return relative / distance


def simulate_camera_observation(true_direction, sigma_degrees):
    """Perturb a unit direction by normally distributed angular noise."""
    if sigma_degrees < 0:
        raise ValueError("Angular uncertainty cannot be negative.")

    direction = np.asarray(true_direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    sigma = np.radians(sigma_degrees)

    tangent = np.random.normal(size=3)
    tangent -= np.dot(tangent, direction) * direction
    tangent_norm = np.linalg.norm(tangent)
    while tangent_norm == 0:
        tangent = np.random.normal(size=3)
        tangent -= np.dot(tangent, direction) * direction
        tangent_norm = np.linalg.norm(tangent)
    tangent /= tangent_norm

    angle = np.random.normal(loc=0.0, scale=sigma)
    observed = np.cos(angle) * direction + np.sin(angle) * tangent
    return observed / np.linalg.norm(observed)


def select_objects(number_of_objects):
    """Select evenly spaced objects from the 20-object synthetic dataset."""
    if not 1 <= number_of_objects <= len(CELESTIAL_OBJECTS):
        raise ValueError("Requested observation count is outside the dataset.")
    items = list(CELESTIAL_OBJECTS.items())
    indices = np.linspace(0, len(items) - 1, number_of_objects, dtype=int)
    return {items[i][0]: items[i][1] for i in indices}


# ------------------------------ Estimation ----------------------------------
def calculate_residuals(estimated_position, objects, observations):
    """Concatenate observed-minus-predicted 3D direction-vector residuals."""
    residuals = []
    for name, observed in observations.items():
        predicted = calculate_direction(objects[name], estimated_position)
        residuals.extend(observed - predicted)
    return np.asarray(residuals)


def estimate_position(objects, observations):
    """Estimate the three spacecraft coordinates using SciPy least_squares."""
    residual_function = lambda position: calculate_residuals(
        position, objects, observations
    )
    return least_squares(residual_function, INITIAL_GUESS)


def run_navigation_trial(number_of_objects, sigma_degrees):
    """Generate observations, estimate position, and return 3D position error."""
    objects = select_objects(number_of_objects)
    observations = {}
    for name, object_position in objects.items():
        true_direction = calculate_direction(object_position, TRUE_POSITION)
        observations[name] = simulate_camera_observation(
            true_direction, sigma_degrees
        )

    result = estimate_position(objects, observations)
    error = float(np.linalg.norm(result.x - TRUE_POSITION))
    return result.x, error, bool(result.success)


# -------------------------------- Experiments -------------------------------
def run_experiment_1():
    """Vary object count while holding angular uncertainty at 0.05 degrees."""
    results = {}
    print("\nEXPERIMENT 1 — Number of celestial observations")
    for count in OBSERVATION_COUNTS:
        errors = []
        for _ in range(TRIALS_PER_CONDITION):
            _, error, success = run_navigation_trial(count, EXPERIMENT_1_SIGMA)
            if success:
                errors.append(error)
        if not errors:
            raise RuntimeError(f"No successful trials for {count} observations.")
        results[count] = errors
        print(
            f"{count:2d} objects | successful: {len(errors):3d}/100 | "
            f"mean: {np.mean(errors):.6f} | SD: {np.std(errors, ddof=1):.6f}"
        )
    return results


def run_experiment_2():
    """Vary angular uncertainty while holding the observation count at 20."""
    results = {}
    print("\nEXPERIMENT 2 — Angular measurement uncertainty")
    for sigma in UNCERTAINTY_LEVELS:
        errors = []
        for _ in range(TRIALS_PER_CONDITION):
            _, error, success = run_navigation_trial(20, sigma)
            if success:
                errors.append(error)
        if not errors:
            raise RuntimeError(f"No successful trials for sigma={sigma}.")
        results[sigma] = errors
        print(
            f"{sigma:5.2f} degrees | successful: {len(errors):3d}/100 | "
            f"mean: {np.mean(errors):.6f} | SD: {np.std(errors, ddof=1):.6f}"
        )
    return results


# -------------------------- Mathematical diagnostics -----------------------
def run_mathematical_validation():
    """Record Jacobian rank and singular values for each observation count."""
    rows = []
    print("\nMATHEMATICAL VALIDATION")
    for count in OBSERVATION_COUNTS:
        objects = select_objects(count)
        observations = {}
        for name, object_position in objects.items():
            direction = calculate_direction(object_position, TRUE_POSITION)
            observations[name] = simulate_camera_observation(direction, 0.05)

        result = estimate_position(objects, observations)
        singular_values = np.linalg.svd(result.jac, compute_uv=False)
        rank = int(np.linalg.matrix_rank(result.jac))
        rows.append({
            "number_of_observations": count,
            "jacobian_rank": rank,
            "singular_values": ";".join(f"{v:.12g}" for v in singular_values),
            "smallest_singular_value": float(np.min(singular_values)),
            "optimizer_success": bool(result.success),
        })
        print(
            f"{count:2d} objects | Jacobian rank: {rank} | "
            f"singular values: {np.array2string(singular_values, precision=6)}"
        )
    return rows


# ----------------------------- Output utilities -----------------------------
def summarize(experiment_name, results):
    """Create one summary row per experimental condition."""
    rows = []
    for parameter, errors in results.items():
        rows.append({
            "experiment": experiment_name,
            "parameter_value": parameter,
            "successful_trials": len(errors),
            "excluded_trials": TRIALS_PER_CONDITION - len(errors),
            "mean_position_error": float(np.mean(errors)),
            "standard_deviation": float(np.std(errors, ddof=1)),
        })
    return rows


def save_errorbar_plot(results, title, xlabel, filename):
    """Save a 300-dpi mean-error plot with plus/minus one SD error bars."""
    x = list(results)
    means = [np.mean(results[value]) for value in x]
    deviations = [np.std(results[value], ddof=1) for value in x]

    plt.figure(figsize=(8, 5))
    plt.errorbar(x, means, yerr=deviations, marker="o", capsize=5)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Mean 3D Position Error (simulation units)")
    plt.xticks(x)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def write_csv(path, rows, fieldnames):
    """Write dictionaries to a UTF-8 CSV file."""
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_outputs(experiment_1, experiment_2, validation_rows):
    """Save experimental summaries, diagnostics, and both research figures."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = (
        summarize("observation_count", experiment_1)
        + summarize("angular_uncertainty_degrees", experiment_2)
    )
    write_csv(
        RESULTS_DIR / "experimental_results.csv",
        summary_rows,
        ["experiment", "parameter_value", "successful_trials",
         "excluded_trials", "mean_position_error", "standard_deviation"],
    )
    write_csv(
        RESULTS_DIR / "mathematical_validation.csv",
        validation_rows,
        ["number_of_observations", "jacobian_rank", "singular_values",
         "smallest_singular_value", "optimizer_success"],
    )

    save_errorbar_plot(
        experiment_1,
        "Effect of Observation Count on Navigation Error",
        "Number of Celestial Objects Observed",
        "experiment_1_observation_count.png",
    )
    save_errorbar_plot(
        experiment_2,
        "Effect of Angular Uncertainty on Navigation Error",
        "Angular Uncertainty (degrees)",
        "experiment_2_angular_uncertainty.png",
    )


def preserve_original_random_stream():
    """
    Retain the 46 preliminary observation draws in the submitted notebook.

    The original Colab script generated 46 exploratory observations before
    the two formal experiments. Replaying those random draws keeps the formal
    experiments on the same seeded random stream as the submitted script.
    """
    for _ in range(46):
        simulate_camera_observation(np.array([1.0, 0.0, 0.0]), 0.05)


def main():
    """Run both experiments, diagnostics, and all file exports."""
    np.random.seed(RANDOM_SEED)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("AUTONOMOUS SPACECRAFT NAVIGATION SIMULATION")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Synthetic celestial objects: {len(CELESTIAL_OBJECTS)}")
    print(f"Trials per experimental condition: {TRIALS_PER_CONDITION}")

    preserve_original_random_stream()
    experiment_1 = run_experiment_1()
    experiment_2 = run_experiment_2()
    validation = run_mathematical_validation()
    save_outputs(experiment_1, experiment_2, validation)

    successful = sum(map(len, experiment_1.values()))
    successful += sum(map(len, experiment_2.values()))
    print("\nRUN COMPLETE")
    print(f"Successful experimental trials: {successful}/1000")
    print(f"Plots and CSV files saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
