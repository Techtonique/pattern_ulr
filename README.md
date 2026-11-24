# pattern_ulr

Ultimate loss ratio (ULR) distributions for an immature book, using the development
pattern of a mature (legacy) book.

- `MackChainladder`: Mack (1993) standard errors, process/parameter split, portfolio
  total, lognormal ULR distribution.
- `BootChainladder`: ODP bootstrap (England & Verrall 2002), simulated ULRs.

## Install

```bash
pip install .            # or: python setup.py install
pip install .[test]      # adds pytest + chainladder (validation tests only; needs numba)
pip install .[examples]  # adds matplotlib for examples/demo.py
```

## Input

One long DataFrame per book (column names configurable on each class):

| origin | dev | cumulative_loss | earned_premium |
|---|---|---|---|
| 2020 | 12 | 1000 | 5000 |
| 2020 | 24 | 1800 | 5000 |

`dev` is an integer age on the same grid in both books. `earned_premium` is needed
for the new book; with it in the legacy book too, `MackChainladder().fit(legacy_df).predict(legacy_df)`
gives legacy ULRs to compare against.

## Usage

Runnable as is: needs only pandas and this package.

```python
import pandas as pd
from pattern_ulr import MackChainladder, BootChainladder, volume_ratio

# Legacy (mature) book: State Farm commercial auto, net paid loss by accident year,
# ages 12, 24, ... 120 months (NAIC Schedule P, via the CAS loss reserving database)
legacy_paid = {
    1988: [54699, 108337, 143899, 164818, 179538, 185391, 188023, 189759, 190520, 193499],
    1989: [60091, 119366, 151151, 174665, 185469, 192213, 196152, 198013, 199997],
    1990: [65860, 130803, 172390, 197977, 210230, 219267, 222428, 224078],
    1991: [61946, 121108, 158880, 182689, 195247, 201854, 204911],
    1992: [65043, 128550, 164433, 187508, 199823, 208008],
    1993: [72295, 144579, 185446, 208388, 219345],
    1994: [81988, 151197, 189630, 212446],
    1995: [83207, 152470, 190974],
    1996: [79699, 143590],
    1997: [75827],
}
legacy_premium = {1988: 286378, 1989: 308908, 1990: 326503, 1991: 332616, 1992: 341890,
                  1993: 355840, 1994: 379781, 1995: 398755, 1996: 406609, 1997: 406516}
legacy_df = pd.DataFrame(
    [(origin, 12 * (k + 1), loss, legacy_premium[origin])
     for origin, losses in legacy_paid.items() for k, loss in enumerate(losses)],
    columns=["origin", "dev", "cumulative_loss", "earned_premium"],
)

# New (immature) book: one row per origin x observed age
new_df = pd.DataFrame({
    "origin":          [2020] * 5 + [2021] * 4 + [2022] * 3 + [2023] * 2 + [2024],
    "dev":             [12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    "cumulative_loss": [7052, 13214, 15641, 19904, 20735, 7048, 13391, 17138, 19272,
                        7483, 14789, 18693, 8492, 16040, 8326],
    "earned_premium":  [30000] * 5 + [32000] * 4 + [34000] * 3 + [37000] * 2 + [40000],
})

# Rescale process variance to the new book's size (premium, since both books have it)
scale = volume_ratio(new_df, legacy_df, size_col="earned_premium")

mack = MackChainladder().fit(legacy_df)
print(mack.predict(new_df, process_scale=scale))   # per origin + Total: SEs, ULR percentiles
print(mack.ulr_distribution("Total").ppf(0.995))   # 99.5th percentile of the total ULR

boot = BootChainladder(n_sims=10_000).fit(legacy_df)
boot.predict(new_df, process_scale=scale)           # simulated ULRs, shape (n_sims, origins + Total)
print(boot.summary())                               # same columns as Mack, for comparison

# Legacy ULRs, to compare with the new book
print(MackChainladder().fit(legacy_df).predict(legacy_df)[["ulr_mean", "ulr_p05", "ulr_p95"]])
```

## Example

`python examples/demo.py` borrows the pattern of a real mature book (State Farm
commercial auto, net paid loss and net earned premium, 1988-97, stored in
`examples/data/comauto_legacy.csv`), projects a synthetic 5-year book with both models, and compares legacy
and new ULRs. It saves three charts:

- `mack_ulr.png`: pattern, projections, ULR intervals, process/parameter split;
- `bootstrap_vs_mack.png`: simulated vs lognormal ULR per origin;
- `legacy_vs_new_ulr.png`: legacy vs new book ULR, origin by origin.

## Tests

`pytest tests/` checks both classes against `chainladder` on its `genins` sample.