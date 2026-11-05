"""House-style colours, as in the R package (R/colors.R)."""
from __future__ import annotations

import numpy as np

ta_navy = "#00365E"
ta_blue = "#0073AB"
ta_lightblue = "#A8C8E0"
ta_gold = "#D39F27"
ta_muted = "#6B7A8D"
ta_years_base = ["#00365E", "#0073AB", "#1A8FC2", "#4AADD4",
                 "#A8C8E0", "#D39F27", "#6B7A8D", "#C44536", "#8B5E3C"]


def ta_year_palette(n):
    """n distinct colours: the nine base colours, interpolated beyond nine
    the way R's colorRampPalette() does."""
    n = int(n)
    if n <= len(ta_years_base):
        return ta_years_base[:n]
    rgb = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                    for c in ta_years_base], dtype=float)
    pos = np.linspace(0, 1, len(ta_years_base))
    out = []
    for t in np.linspace(0, 1, n):
        v = [np.interp(t, pos, rgb[:, k]) for k in range(3)]
        out.append("#%02X%02X%02X" % tuple(int(np.floor(x + 0.5)) for x in v))
    return out
