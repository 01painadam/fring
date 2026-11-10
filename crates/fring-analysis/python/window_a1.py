"""Window lambda with the definition of plots.py window_summary (A1): the median of
the per-bin median MEGNO of the 2 km bins that lie fully inside the window, and
lambda = 2 median / years. Usage: python window_a1.py <profile.csv> <years>
(profile.csv from `fring-analyse map ... 139700 141100 7 10 2`). Lost particles are not in
the profile (class Invalid), so this is the survivors, as in `fring-analyse windows`."""
import sys
import pandas as pd

WINS = [("inner edge", 139700, 139950), ("inner flank", 139950, 140150),
        ("core", 140197, 140247), ("quiet zone", 140150, 140700),
        ("outer flank", 140700, 140900), ("outer edge to 141,050", 140900, 141050),
        ("outer edge to 141,100", 140900, 141100)]
p = pd.read_csv(sys.argv[1])
yrs = float(sys.argv[2])
for name, lo, hi in WINS:
    q = p[(p.r_lo_km >= lo) & (p.r_hi_km <= hi)]
    if q.empty or q.n.sum() < 500:
        continue
    m = round(float(q.median_megno.median()), 1)
    print(f"{name},{int(q.n.sum())},{m},{round(2 * m / yrs, 3)},{2 * float(q.median_megno.median()) / yrs:.4f}")
