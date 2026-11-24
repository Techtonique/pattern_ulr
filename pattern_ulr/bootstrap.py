"""ODP bootstrap chain ladder (England & Verrall 2002) on a borrowed pattern."""
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .helpers import to_triangle, latest_loss_and_premium


class BootChainladder:
    """
    Over-dispersed Poisson (ODP) bootstrap chain ladder: simulated ULR distributions.

    ``fit`` fits the ODP chain ladder to the legacy book, then resamples its Pearson
    residuals ``n_sims`` times and refits the link ratios each time (parameter risk).
    ``predict`` projects the new book with each simulated pattern, drawing each
    future incremental from a gamma distribution (process risk). Origins in the same
    simulation share one pattern, so they are correlated through it.

    Parameters
    ----------
    n_sims : int, default 10_000
        Number of bootstrap simulations.
    chunk : int, default 1_000
        Simulations per vectorised batch. Memory ~ chunk * n_origins * n_devs * 8 bytes.
    seed : int, default 0
        Random seed for resampling and process draws.
    origin_col, dev_col, loss_col, premium_col : str
        Column names, as in ``MackChainladder``.

    Attributes
    ----------
    phi_ : float
        ODP scale parameter, in legacy units. Set by ``fit``.
    link_ratio_sims_ : pandas.DataFrame
        Shape (n_sims, n_devs - 1): simulated patterns, columns by "from" age.
    ultimates_, ulrs_ : pandas.DataFrame
        Shape (n_sims, n_origins + 1): simulated ultimates and ULRs, one column per
        origin plus "Total". Set by ``predict``.

    Notes
    -----
    No hat-matrix adjustment is applied to residuals; this matches chainladder's
    ``BootstrapODPSample(hat_adj=False)``.

    Examples
    --------
    >>> model = BootChainladder(n_sims=10_000).fit(legacy_df)
    >>> ulrs = model.predict(new_df, process_scale=volume_ratio(new_df, legacy_df))
    >>> model.summary()
    """

    PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

    def __init__(self, n_sims=10_000, chunk=1_000, seed=0, origin_col="origin", dev_col="dev",
                 loss_col="cumulative_loss", premium_col="earned_premium"):
        self.n_sims, self.chunk, self.seed = n_sims, chunk, seed
        self.origin_col, self.dev_col = origin_col, dev_col
        self.loss_col, self.premium_col = loss_col, premium_col

    def fit(self, legacy_df):
        """
        Fit the ODP chain ladder to the legacy book and bootstrap its link ratios.

        Steps:

        1. Volume-weighted link ratios f_k.
        2. Fitted cumulatives, backward from each latest diagonal: C(k) = C(k+1) / f_k.
        3. Unscaled Pearson residuals of incrementals, (I - m) / sqrt(|m|).
        4. Scale phi = sum(r^2) / (n - p), with p = 2 * n_origins - 1.
        5. Residuals adjusted by sqrt(n / (n - p)); zero residuals (corners) dropped.
        6. For each simulation: resample residuals, build pseudo-incrementals
           m + r* sqrt(|m|), cumulate, refit volume-weighted link ratios.

        Parameters
        ----------
        legacy_df : pandas.DataFrame
            Long-format legacy data.

        Returns
        -------
        BootChainladder
            ``self``, with ``phi_`` and ``link_ratio_sims_`` set.
        """
        triangle = to_triangle(legacy_df, self.origin_col, self.dev_col, self.loss_col)
        cumulative = triangle.to_numpy(float)
        observed = ~np.isnan(cumulative)
        n_origins, n_devs = cumulative.shape

        link_ratio = np.array([np.nansum(np.where(observed[:, j + 1], cumulative[:, j + 1], np.nan)) /
                               np.nansum(np.where(observed[:, j + 1], cumulative[:, j], np.nan))
                               for j in range(n_devs - 1)])
        fitted_cumulative = np.full_like(cumulative, np.nan)
        for i in range(n_origins):
            last = observed[i].nonzero()[0][-1]
            fitted_cumulative[i, last] = cumulative[i, last]
            for j in range(last - 1, -1, -1):
                fitted_cumulative[i, j] = fitted_cumulative[i, j + 1] / link_ratio[j]

        incremental = np.diff(cumulative, axis=1, prepend=0.0)
        fitted_incremental = np.diff(fitted_cumulative, axis=1, prepend=0.0)
        pearson = (incremental - fitted_incremental) / np.sqrt(np.abs(fitted_incremental))
        n_obs, n_params = observed.sum(), 2 * n_origins - 1
        self.phi_ = np.nansum(pearson**2) / (n_obs - n_params)
        adjusted = pearson * np.sqrt(n_obs / (n_obs - n_params))
        residual_pool = adjusted[observed & (np.abs(adjusted) > 1e-12)]

        self.link_ratio_sims_ = self._resample_link_ratios(fitted_incremental, observed, residual_pool,
                                                           triangle.columns[:-1])
        return self

    def predict(self, new_df, process_scale=1.0):
        """
        Simulate ultimate losses and ULRs for the new book.

        Each origin is projected from its latest diagonal, one age at a time, with
        each simulated pattern. Each future incremental is drawn from
        gamma(mean m, variance phi_new * m), with m = C * (f - 1) and
        phi_new = phi_ * process_scale.

        Parameters
        ----------
        new_df : pandas.DataFrame
            Long-format new-book data with loss and premium columns. Every age in it
            must also appear in the legacy pattern.
        process_scale : float, default 1.0
            Multiplier on phi, typically ``volume_ratio(new_df, legacy_df)``. Use 1.0
            when ``new_df`` is the legacy book itself.

        Returns
        -------
        pandas.DataFrame
            ``ulrs_``: simulated ULRs, shape (n_sims, n_origins + 1). Simulated
            ultimates are kept in ``ultimates_``.

        Raises
        ------
        ValueError
            If the loss and premium triangles differ.

        Notes
        -----
        Expected increments are floored at 1e-9 because the gamma needs a positive
        mean, so link ratios below 1 are treated as zero development.
        """
        rng = np.random.default_rng(self.seed + 1)
        latest_age, latest_loss, latest_premium = latest_loss_and_premium(
            new_df, self.origin_col, self.dev_col, self.loss_col, self.premium_col)
        phi_new = self.phi_ * process_scale
        sims = self.link_ratio_sims_

        ultimates = {}
        for origin in tqdm(latest_age.index, desc="project new book"):
            future_f = sims.loc[:, sims.columns >= latest_age[origin]].to_numpy()
            cumulative = np.full(len(future_f), float(latest_loss[origin]))
            for step in range(future_f.shape[1]):
                expected_increment = np.maximum(cumulative * (future_f[:, step] - 1), 1e-9)
                cumulative = cumulative + rng.gamma(expected_increment / phi_new, phi_new)
            ultimates[origin] = cumulative

        self.ultimates_ = pd.DataFrame(ultimates)
        self.ultimates_["Total"] = self.ultimates_.sum(axis=1)
        premium = pd.concat([latest_premium, pd.Series({"Total": latest_premium.sum()})])
        self.ulrs_ = self.ultimates_ / premium
        return self.ulrs_

    def summary(self):
        """
        Summary statistics of the simulated ultimates and ULRs.

        Returns
        -------
        pandas.DataFrame
            One row per origin plus "Total", with columns ``ultimate``, ``total_se``
            (standard deviation of the ultimate), ``ulr_mean`` and ``ulr_p05`` ...
            ``ulr_p95``, named as in ``MackChainladder.predict`` for easy comparison.
        """
        out = pd.DataFrame({"ultimate": self.ultimates_.mean(), "total_se": self.ultimates_.std(),
                            "ulr_mean": self.ulrs_.mean()})
        for q in self.PERCENTILES:
            out[f"ulr_p{int(q * 100):02d}"] = self.ulrs_.quantile(q)
        return out

    def _resample_link_ratios(self, fitted_incremental, observed, residual_pool, ages):
        """Refit volume-weighted link ratios on n_sims pseudo-triangles, in chunks."""
        rng = np.random.default_rng(self.seed)
        pair = observed[:, 1:]
        sims = []
        for start in tqdm(range(0, self.n_sims, self.chunk), desc="bootstrap legacy pattern"):
            size = min(self.chunk, self.n_sims - start)
            residuals = rng.choice(residual_pool, size=(size, *fitted_incremental.shape))
            pseudo = np.where(observed, fitted_incremental + residuals * np.sqrt(np.abs(fitted_incremental)), 0.0)
            pseudo_cumulative = np.cumsum(pseudo, axis=2)
            numerator = np.where(pair, pseudo_cumulative[:, :, 1:], 0).sum(axis=1)
            denominator = np.where(pair, pseudo_cumulative[:, :, :-1], 0).sum(axis=1)
            sims.append(numerator / denominator)
        return pd.DataFrame(np.vstack(sims), columns=ages)
