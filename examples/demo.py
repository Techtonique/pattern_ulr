"""
End-to-end example: borrow a mature book's pattern, project a small new book with Mack
and the ODP bootstrap, compare legacy and new ULRs, and chart the results.

Legacy book: State Farm commercial auto, net paid loss and net earned premium,
accident years 1988-1997 (NAIC Schedule P data, from the CAS loss reserving database),
stored in ``examples/data/comauto_legacy.csv``. No chainladder dependency.
New book: synthetic, 5 accident years written at a worse ULR than the legacy book.

Run from anywhere after ``pip install .[examples]``:

    python examples/demo.py

Writes ``mack_ulr.png``, ``bootstrap_vs_mack.png`` and ``legacy_vs_new_ulr.png`` to the
current directory.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pattern_ulr import BootChainladder, MackChainladder, to_triangle, volume_ratio

# ----------------------------------------------------------------------
# 1. Data prep (external to the package): long tables
# ----------------------------------------------------------------------
legacy_df = pd.read_csv(Path(__file__).parent / "data" / "comauto_legacy.csv")
print("legacy_df", legacy_df.head(3).to_string(index=False), "\n")

mack = MackChainladder().fit(legacy_df)

# Synthetic new book: 5 accident years at ~72% ULR (legacy runs at ~65%),
# developed along the legacy pattern, about a tenth of the legacy book's size
rng = np.random.default_rng(1)
cdf_from_age = mack.params_.link_ratio[::-1].cumprod()[::-1]
premium_by_origin = {2020: 30_000, 2021: 32_000, 2022: 34_000, 2023: 37_000, 2024: 40_000}
rows = []
for k, (origin, premium) in enumerate(premium_by_origin.items()):
    true_ultimate = premium * 0.72 * rng.lognormal(0, 0.06)
    for age in mack.params_.index[: 5 - k]:
        rows.append(dict(origin=origin, dev=age, earned_premium=premium,
                         cumulative_loss=round(true_ultimate / cdf_from_age[age] * rng.lognormal(0, 0.05))))
new_df = pd.DataFrame(rows)
print("new_df", new_df.head(6).to_string(index=False), "\n")

# ----------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------
# Book size from premium, now that both books have it
scale = volume_ratio(new_df, legacy_df, size_col="earned_premium")
mack_results = mack.predict(new_df, process_scale=scale)

# Legacy ULRs: the same model applied to the legacy book itself (scale 1)
legacy_results = MackChainladder().fit(legacy_df).predict(legacy_df)

boot = BootChainladder(n_sims=10_000).fit(legacy_df)
boot_ulrs = boot.predict(new_df, process_scale=scale)

cols = ["ultimate", "total_se", "ulr_p05", "ulr_p50", "ulr_p95"]
print(f"process scale = {scale:.2e}\n")
print("Mack (lognormal)\n", mack_results[cols].round(3), "\n")
print("Bootstrap\n", boot.summary()[cols].round(3), "\n")
print("Legacy book (Mack)\n", legacy_results[cols].round(3), "\n")
print(f"Total ULR 99.5th: Mack {mack.ulr_distribution('Total').ppf(0.995):.1%} | "
      f"bootstrap {boot_ulrs['Total'].quantile(0.995):.1%}")

# ----------------------------------------------------------------------
# 3. Charts
# ----------------------------------------------------------------------
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
percent = plt.FuncFormatter(lambda v, _: f"{v:.0%}")
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "axes.titleweight": "bold", "axes.titlesize": 11, "axes.titlelocation": "left",
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
origins = mack_results.drop(index="Total")
params = mack.params_

fig, ax = plt.subplots(2, 2, figsize=(13, 9.5))

# A. Legacy link ratios: individual vs selected, 95% parameter band
a = ax[0, 0]
legacy_triangle = to_triangle(legacy_df, "origin", "dev", "cumulative_loss")
individual = (legacy_triangle.shift(-1, axis=1) / legacy_triangle).iloc[:, :-1]
x = np.arange(len(params))
for j, age in enumerate(params.index):
    v = individual[age].dropna()
    a.scatter(np.full(len(v), j) + rng.uniform(-.12, .12, len(v)), v, s=14, color="#b5b4ae", zorder=2)
a.errorbar(x, params.link_ratio, yerr=1.96 * np.sqrt(params.sigma_sq / params.volume),
           fmt="o", color=COLORS[0], ms=7, capsize=4, lw=2, zorder=3)
a.set_yscale("log"); a.yaxis.set_minor_locator(plt.NullLocator())
ticks = [t for t in (1, 1.1, 1.25, 1.5, 2, 3, 5) if t <= individual.max().max() * 1.05]
a.set_yticks(ticks); a.set_yticklabels([f"{t:g}" for t in ticks])
a.set_xticks(x, [f"{d}–{d + 12}" for d in params.index], rotation=45)
a.set_title("A · Legacy link ratios: individual (gray), selected ±95% band")
a.set_ylabel("age-to-age factor (log)")

# B. New book loss ratio: actual, projected, 5–95% at ultimate
b = ax[0, 1]
new_triangle = to_triangle(new_df, "origin", "dev", "cumulative_loss")
ultimate_age = params.index[-1] + 12
for k, (origin, r) in enumerate(origins.iterrows()):
    observed = new_triangle.loc[origin].dropna() / r.premium
    future = params[params.index >= r.latest_age]
    projected = r.latest_loss * future.link_ratio.cumprod() / r.premium
    b.plot(observed.index, observed.values, color=COLORS[k], lw=2, marker="o", ms=5)
    b.plot([r.latest_age, *(future.index + 12)], [observed.iloc[-1], *projected.values],
           color=COLORS[k], lw=1.5, ls="--")
    b.vlines(ultimate_age + 4 + 3 * k, r.ulr_p05, r.ulr_p95, color=COLORS[k], lw=3)
    b.annotate(str(origin), (observed.index[-1], observed.iloc[-1]), xytext=(4, -10),
               textcoords="offset points", color=INK2, fontsize=9)
b.set_title("B · Loss ratio: actual, projected, 5–95% at ultimate")
b.set_xlabel("development age (months)"); b.set_ylabel("loss / earned premium")
b.yaxis.set_major_formatter(percent)

# C. Lognormal ULR intervals per origin + total
c = ax[1, 0]
for k, (origin, r) in enumerate(mack_results.iterrows()):
    color = INK if origin == "Total" else COLORS[k]
    y = len(mack_results) - 1 - k
    c.hlines(y, r.ulr_p05, r.ulr_p95, color=color, lw=2)
    c.hlines(y, r.ulr_p25, r.ulr_p75, color=color, lw=9)
    c.plot(r.ulr_p50, y, "o", color="white", mec=color, mew=2, ms=8, zorder=3)
    c.annotate(f"median {r.ulr_p50:.0%} · 95th {r.ulr_p95:.0%}", (r.ulr_p95, y), xytext=(6, -3),
               textcoords="offset points", fontsize=9, color=INK2)
c.set_yticks(range(len(mack_results))[::-1], [str(o) for o in mack_results.index]); c.grid(axis="y", visible=False)
span = mack_results.ulr_p95.max() - mack_results.ulr_p05.min()
c.set_xlim(mack_results.ulr_p05.min() - 0.05 * span, mack_results.ulr_p95.max() + 0.6 * span); c.xaxis.set_major_formatter(percent)
c.set_title("C · Lognormal ULR: 5–95% (thin), 25–75% (thick), median (dot)")
c.set_xlabel("ultimate loss ratio")

# D. Variance split: process vs parameter
d = ax[1, 1]
process_share = origins.process_se**2 / origins.total_se**2
xs = np.arange(len(origins))
d.bar(xs, process_share, color=COLORS[0], width=0.6, label="process (randomness of claims)")
d.bar(xs, 1 - process_share, bottom=process_share, color=COLORS[1], width=0.6,
      label="parameter (borrowed pattern)", edgecolor=SURFACE, linewidth=2)
for i, cv in enumerate(origins.total_se / origins.ultimate):
    d.text(i, 1.02, f"CV {cv:.0%}", ha="center", fontsize=9, color=INK2)
d.set_xticks(xs, [f"{o}\n@{a:.0f}m" for o, a in zip(origins.index, origins.latest_age)])
d.set_ylim(0, 1.22); d.yaxis.set_major_formatter(percent); d.grid(axis="x", visible=False)
d.set_title("D · Share of ultimate variance: process vs parameter")
d.legend(loc="upper center", ncol=2, frameon=False, fontsize=9)

fig.tight_layout(); fig.savefig("mack_ulr.png", dpi=130)

# Bootstrap histogram vs Mack lognormal
fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
grid = np.linspace(boot_ulrs.quantile(0.0005).min(), boot_ulrs.quantile(0.9995).max(), 400)
for k, (axis, origin) in enumerate(zip(axes.ravel(), mack_results.index)):
    color = INK if origin == "Total" else COLORS[k]
    sims = boot_ulrs[origin]
    axis.hist(sims, bins=80, density=True, color=color, alpha=0.35, edgecolor="none", label="bootstrap")
    axis.plot(grid, mack.ulr_distribution(origin).pdf(grid), color=color, lw=2, label="Mack lognormal")
    axis.axvline(sims.quantile(.95), color=color, ls=":", lw=1.5)
    axis.set_title(f"{origin} · 95th: boot {sims.quantile(.95):.0%} vs LN {mack_results.at[origin, 'ulr_p95']:.0%}",
                   fontsize=10)
    axis.set_yticks([]); axis.spines["left"].set_visible(False); axis.grid(False)
    axis.xaxis.set_major_formatter(percent)
axes[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("ULR: bootstrap simulations (histogram) vs Mack lognormal (line); dotted = bootstrap 95th",
             x=0.01, ha="left", fontweight="bold")
fig.tight_layout(); fig.savefig("bootstrap_vs_mack.png", dpi=130)

# Legacy vs new ULR: are the new years running hotter than the book the pattern came from?
both_books = pd.concat([legacy_results.rename(index={"Total": "Legacy total"}),
                        mack_results.rename(index={"Total": "New total"})])
is_new = [str(o) in map(str, mack_results.index.drop("Total")) or o == "New total" for o in both_books.index]
legacy_avg = legacy_results.at["Total", "ulr_mean"]
fig, axis = plt.subplots(figsize=(11, 7.5))
for y, ((origin, r), new) in enumerate(zip(both_books.iloc[::-1].iterrows(), is_new[::-1])):
    color = COLORS[0] if new else "#9a9993"
    weight = 3 if "total" in str(origin) else 2
    axis.hlines(y, r.ulr_p05, r.ulr_p95, color=color, lw=weight)
    axis.hlines(y, r.ulr_p25, r.ulr_p75, color=color, lw=weight + 7)
    axis.plot(r.ulr_p50, y, "o", color="white", mec=color, mew=2, ms=7, zorder=3)
axis.axvline(legacy_avg, color=INK2, ls="--", lw=1)
axis.annotate(f"legacy average {legacy_avg:.0%}", (legacy_avg, len(both_books) - 0.6), xytext=(4, 0),
              textcoords="offset points", fontsize=9, color=INK2)
axis.set_yticks(range(len(both_books)), [str(o) for o in both_books.index[::-1]])
axis.grid(axis="y", visible=False); axis.xaxis.set_major_formatter(percent)
axis.set_xlabel("ultimate loss ratio")
axis.set_title("Legacy (gray) vs new book (blue) ULR, Mack lognormal: 5–95%, 25–75%, median")
fig.tight_layout(); fig.savefig("legacy_vs_new_ulr.png", dpi=130)
print("\nSaved mack_ulr.png, bootstrap_vs_mack.png and legacy_vs_new_ulr.png")