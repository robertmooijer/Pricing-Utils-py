"""Which double-lift verdict each kind of interval gives, by scenario.

The evidence behind RATE-11 in TECHNICAL.md. The statistic and the
bootstrap are those of double_lift(): bins by the new/old ratio, cut once;
mean |A/E - 1| over the bins; Poisson(1) policy weights; the new premium
rebased to the old total. Each scenario draws 300 books with a known truth
and reports, per interval, how often the verdict is new / old /
inconclusive and how often the interval covers the true difference (the
same statistic on the expected claims). The package uses the interval
"around the point value" (recentred).

Run from the project root (about 25 minutes):
    python demo/double_lift_interval_coverage.py
"""
import numpy as np

rng = np.random.default_rng(12)


def bins_of(ratio, w, k=10):
    o = np.argsort(ratio, kind="stable")
    cw = np.cumsum(w[o]) / w.sum()
    b = np.empty(len(ratio), dtype=int)
    b[o] = np.minimum((cw * k - 1e-12).astype(int), k - 1)
    return b


def mad(a, p):
    return np.mean(np.abs(a / p - 1))


def stat(act, pn, po, bin_, k=10):
    A = np.bincount(bin_, act, k)
    return mad(A, np.bincount(bin_, po, k)) - mad(A, np.bincount(bin_, pn, k))


def run(name, n, old_eff, new_eff, reps=300, B=200, conf=.9):
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    w = rng.uniform(.05, 1, n)
    lam = np.exp(-2 + .3 * x1 + .3 * x2)
    old = np.exp(-2 + .3 * x1 + old_eff * x2)
    new = np.exp(-2 + .3 * x1 + new_eff * x2)
    new = new * (old * w).sum() / (new * w).sum()
    bin_ = bins_of(new / old, w)
    pn0, po0 = new * w, old * w
    D = stat(lam * w, pn0, po0, bin_)
    q = [(1 - conf) / 2, 1 - (1 - conf) / 2]
    res = {m: {"new": 0, "old": 0, "inconcl": 0, "cover": 0}
           for m in ("percentile", "recentred", "basic")}
    for _ in range(reps):
        act = rng.poisson(lam * w)
        th = stat(act, pn0, po0, bin_)
        sims = np.empty(B)
        for b in range(B):
            kk = rng.poisson(1, n)
            pn, po = pn0 * kk, po0 * kk
            pn = pn * po.sum() / pn.sum()
            sims[b] = stat(act * kk, pn, po, bin_)
        ql, qh = np.quantile(sims, q)
        m = sims.mean()
        for meth, (a, c) in {"percentile": (ql, qh),
                             "recentred": (th + ql - m, th + qh - m),
                             "basic": (2 * th - qh, 2 * th - ql)}.items():
            v = "new" if a > 0 else "old" if c < 0 else "inconcl"
            res[meth][v] += 1
            res[meth]["cover"] += a <= D <= c
    truth = "new" if D > 0 else "old"
    print("%-40s D=%+.4f (better: %s)" % (name, D, truth))
    for meth, r in res.items():
        print("   %-10s new %5.1f%%  old %5.1f%%  inconclusive %5.1f%%  coverage %5.1f%%"
              % (meth, *(100 * r[k] / reps for k in ("new", "old", "inconcl", "cover"))))


run("new right, old misses half, 40k", 40000, .15, .30)
run("new right, old misses a sixth, 40k", 40000, .25, .30)
run("mirrored errors (D ~ 0), 40k", 40000, .24, .36)
run("old better, 40k", 40000, .2, .4)
run("new right, old misses half, 10k", 10000, .15, .30)
run("mirrored errors (D ~ 0), 10k", 10000, .24, .36)
