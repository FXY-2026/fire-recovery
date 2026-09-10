import numpy as np

def is_growing_season(month):
    return month >= 10 or month <= 4

def get_recovery_window(gpp_tmin, gpp_rec, et_tmin, et_rec):
    gpp_end = gpp_tmin + gpp_rec
    et_end = et_tmin + et_rec
    if gpp_end >= et_end:
        return gpp_tmin, gpp_end
    else:
        return et_tmin, et_end

def mean_e_et_during_recovery(
    e_et_series,
    months,
    gpp_tmin,
    gpp_rec,
    et_tmin,
    et_rec,
    precipitation,
    forest_mask,
    pr_classes
):
    n_pixels = len(e_et_series)
    mean_values = np.full(
        n_pixels,
        np.nan,
        dtype=np.float32
    )
    for j in range(n_pixels):
        start, end = get_recovery_window(
            gpp_tmin[j],
            gpp_rec[j],
            et_tmin[j],
            et_rec[j]
        )
        start = int(start)
        end = int(end)
        growing_season_e_et = []
        for i in range(len(e_et_series[j])):
            month = months[j][i]
            if is_growing_season(month):
                growing_season_e_et.append(
                    e_et_series[j][i]
                )
        values = []
        for k in range(start + 1, end + 1):
            index = k - 1
            value = growing_season_e_et[index]
            if (
                not np.isnan(value) and
                not np.isinf(value)
            ):
                values.append(value)
        if len(values) > 0:
            mean_values[j] = np.mean(values)
    results_all = {}
    results_forest = {}
    results_nonforest = {}
    for pr_name, class_info in pr_classes.items():
        pr_min = class_info[0]
        pr_max = class_info[1]
        valid = (
            ~np.isnan(mean_values) &
            ~np.isnan(precipitation)
        )
        pr_mask = (
            valid &
            (precipitation >= pr_min) &
            (precipitation < pr_max)
        )
        values_all = mean_values[
            pr_mask
        ]
        values_forest = mean_values[
            pr_mask & forest_mask
        ]
        values_nonforest = mean_values[
            pr_mask & (~forest_mask)
        ]
        results_all[pr_name] = np.asarray(
            values_all,
            dtype=np.float32
        )
        results_forest[pr_name] = np.asarray(
            values_forest,
            dtype=np.float32
        )
        results_nonforest[pr_name] = np.asarray(
            values_nonforest,
            dtype=np.float32
        )
    return (
        mean_values,
        results_all,
        results_forest,
        results_nonforest
    )