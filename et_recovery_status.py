import numpy as np


def classify_recovery_status(
    et_rec,
    t_rec,
    e_rec,
    et_min,
    t_min,
    e_min,
    precipitation,
    forest_mask,
    pr_classes
):
    et_end = et_min + et_rec
    t_end = t_min + t_rec
    e_end = e_min + e_rec
    valid = (
        (et_rec > 0) &
        ((t_rec > 0) | (e_rec > 0))
    )
    results = {}
    for pr_name, (pr_min, pr_max) in pr_classes.items():
        pr_mask = (
            valid &
            (precipitation >= pr_min) &
            (precipitation < pr_max)
        )
        results[pr_name] = {}
        for group, group_mask in {
            "forest": forest_mask,
            "nonforest": ~forest_mask,
            "all": np.ones_like(
                forest_mask,
                dtype=bool
            )
        }.items():
            mask = pr_mask & group_mask
            t_recovered = (
                (t_rec[mask] > 0) &
                (t_end[mask] <= et_end[mask])
            )
            e_recovered = (
                (e_rec[mask] > 0) &
                (e_end[mask] <= et_end[mask])
            )
            counts = {
                "T_recovered_E_not": np.sum(
                    t_recovered & ~e_recovered
                ),
                "E_recovered_T_not": np.sum(
                    ~t_recovered & e_recovered
                ),
                "Both_recovered": np.sum(
                    t_recovered & e_recovered
                )
            }
            results[pr_name][group] = counts
    return results

def initialize_accumulated_results(pr_classes):
    accumulated = {}
    for pr_name in pr_classes:
        accumulated[pr_name] = {}
        for group in [
            "forest",
            "nonforest",
            "all"
        ]:
            accumulated[pr_name][group] = {
                "T_recovered_E_not": 0,
                "E_recovered_T_not": 0,
                "Both_recovered": 0
            }
    return accumulated

def accumulate_results(
    accumulated,
    event_results
):
    for pr_name in event_results:
        for group in event_results[pr_name]:
            for status in event_results[pr_name][group]:
                accumulated[pr_name][group][status] += (
                    event_results[pr_name][group][status]
                )

def calculate_fractions(accumulated):
    fractions = {}
    for pr_name in accumulated:
        fractions[pr_name] = {}
        for group in accumulated[pr_name]:
            counts = accumulated[pr_name][group]
            total = sum(counts.values())
            if total > 0:
                fractions[pr_name][group] = {
                    key: value / total
                    for key, value in counts.items()
                }
            else:
                fractions[pr_name][group] = {
                    key: np.nan
                    for key in counts
                }
    return fractions