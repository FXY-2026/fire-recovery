import numpy as np

LOOKBACK_MONTHS = 36
POST_GSM = 8
Q_THRESHOLD = 0.10
NODATA = -9999.0

def clean_fire_events(fire_stack, years, months):
    season_ids = np.where(
        months >= 10,
        years,
        np.where(months <= 4, years - 1, -1)
    )
    fire_stack[season_ids == -1] = False
    for r in range(fire_stack.shape[1]):
        for c in range(fire_stack.shape[2]):
            seen = set()
            for t in np.flatnonzero(fire_stack[:, r, c]):
                season = int(season_ids[t])
                if season in seen:
                    fire_stack[t, r, c] = False
                else:
                    seen.add(season)
    return fire_stack


def calculate_event(t_fire, X, A, fire_stack, years, months):
    Xhat = X - A
    rows, cols = np.where(fire_stack[t_fire])
    n = len(rows)
    if n == 0:
        return None

    pre = [
        t_fire - i
        for i in range(1, min(t_fire, LOOKBACK_MONTHS) + 1)
        if months[t_fire - i] >= 10 or months[t_fire - i] <= 4
    ]
    if not pre:
        return None
    BA = np.nanmean(
        A[pre][:, rows, cols],
        axis=0
    )
    Cpre = Xhat[pre][:, rows, cols] + BA
    Cpre[Cpre <= 0] = np.nan
    Cmed = np.nanmedian(Cpre, axis=0)
    threshold = Q_THRESHOLD * Cmed

    post = [
        t
        for t in range(t_fire, len(years))
        if months[t] >= 10 or months[t] <= 4
    ]
    post_X = X[post][:, rows, cols].copy()
    post_C = Xhat[post][:, rows, cols] + BA
    valid = (
        np.isfinite(post_X)
        & np.isfinite(post_C)
        & np.isfinite(Cmed)[None, :]
        & (Cmed[None, :] > 0)
        & (post_C > 0)
        & (post_C >= threshold[None, :])
    )
    post_X[~valid] = np.nan
    post_C[~valid] = np.nan

    loss_n = min(POST_GSM, len(post))
    F = (
        100.0
        * (post_X[:loss_n] - post_C[:loss_n])
        / post_C[:loss_n]
    )

    damage = np.full(n, NODATA, dtype=np.float32)
    time_to_min = np.full(n, NODATA, dtype=np.float32)
    recovery_time = np.full(n, NODATA, dtype=np.float32)
    recovery_rate = np.full(n, NODATA, dtype=np.float32)

    for p in range(n):

        reburned = any(
            fire_stack[post[k], rows[p], cols[p]]
            for k in range(1, loss_n)
        )
        if reburned:
            recovery_time[p] = -3
            continue

        if np.all(np.isnan(F[:, p])):
            continue
        kmin = int(np.nanargmin(F[:, p]))
        fmin = float(F[kmin, p])
        if fmin >= 0:
            recovery_time[p] = -2
            continue
        damage[p] = -fmin
        time_to_min[p] = kmin + 1

        recovered_at = None
        reburned = False

        for pos in range(kmin + 1, len(post)):
            t = post[pos]
            if fire_stack[t, rows[p], cols[p]]:
                reburned = True
                break
            xv = post_X[pos, p]
            cv = post_C[pos, p]
            if not (
                np.isfinite(xv)
                and np.isfinite(cv)
                and cv > 0
            ):
                continue
            if xv >= cv:
                recovered_at = pos
                break

        if reburned:
            recovery_time[p] = -3
        elif recovered_at is None:
            recovery_time[p] = -1
        else:
            MTR = recovered_at - kmin
            recovery_time[p] = MTR
            F_recovery = (
                100.0
                * (
                    post_X[recovered_at, p]
                    - post_C[recovered_at, p]
                )
                / post_C[recovered_at, p]
            )
            recovery_rate[p] = (
                F_recovery - fmin
            ) / MTR

    return {
        "rows": rows,
        "cols": cols,
        "damage": damage,
        "time_to_min": time_to_min,
        "recovery_time": recovery_time,
        "recovery_rate": recovery_rate,
    }