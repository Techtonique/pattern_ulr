"""Mack chain ladder on a borrowed pattern, with a lognormal ULR distribution."""
import numpy as np
import pandas as pd
from scipy import stats

from .helpers import to_triangle, latest_loss_and_premium


class MackChainladder:
    """
    Mack (1993) chain ladder: fit the pattern on a legacy book, project a new book.

    ``fit`` estimates link ratios f_k, Mack's sigma_k^2 and the volumes S_k behind
    each f_k from the legacy book. ``predict`` projects the new book to ultimate,
    splits each origin's standard error into process and parameter parts, adds the
    portfolio total (with Mack's cross term), and moment-matches a lognormal to
    each ultimate to give a ULR distribution.

    Parameters
    ----------
    origin_col : str, default "origin"
        Column holding the origin period.
    dev_col : str, default "dev"
        Column holding the development age (integer).
    loss_col : str, default "cumulative_loss"
        Column holding the cumulative loss.
    premium_col : str, default "earned_premium"
        Column holding the cumulative earned premium (new book only).

    Attributes
    ----------
    params_ : pandas.DataFrame
        Set by ``fit``. Indexed by "from" age, with columns ``link_ratio``,
        ``sigma_sq`` (legacy units) and ``volume`` (S_k).
    results_ : pandas.DataFrame
        Set by ``predict``. One row per origin plus a "Total" row; see ``predict``.

    Examples
    --------
    >>> model = MackChainladder().fit(legacy_df)
    >>> results = model.predict(new_df, process_scale=volume_ratio(new_df, legacy_df))
    >>> model.ulr_distribution("Total").ppf(0.995)
    """

    PERCENTILES = (0.05, 0.25, 0.50, 0.75, 0.95)

    def __init__(self, origin_col="origin", dev_col="dev", loss_col="cumulative_loss",
                 premium_col="earned_premium"):
        self.origin_col, self.dev_col = origin_col, dev_col
        self.loss_col, self.premium_col = loss_col, premium_col

    def fit(self, legacy_df):
        """
        Estimate the development pattern and Mack's variance parameters.

        For each "from" age k:

        * the link ratio f_k is volume-weighted: sum of C(k+1) / sum of C(k), using
          only origins observed at both ages;
        * sigma_k^2 is the weighted variance of the individual link ratios around
          f_k, with weights C(k) and n_k - 1 degrees of freedom;
        * the last sigma^2, which has a single observation, uses Mack's
          extrapolation min(s_{n-1}^4 / s_{n-2}^2, s_{n-2}^2, s_{n-1}^2).

        Parameters
        ----------
        legacy_df : pandas.DataFrame
            Long-format legacy data. Needs at least four development ages.

        Returns
        -------
        MackChainladder
            ``self``, with ``params_`` set.
        """
        loss_triangle = to_triangle(legacy_df, self.origin_col, self.dev_col, self.loss_col)
        next_loss = loss_triangle.shift(-1, axis=1)
        both_observed = loss_triangle.notna() & next_loss.notna()
        current, following = loss_triangle[both_observed], next_loss[both_observed]

        volume = current.sum().iloc[:-1]
        link_ratio = following.sum().iloc[:-1] / volume
        n_pairs = both_observed.sum().iloc[:-1]
        sigma_sq = (current * (following / current - link_ratio) ** 2).sum().iloc[:-1] / (n_pairs - 1)
        sigma_sq.iloc[-1] = min(sigma_sq.iloc[-2] ** 2 / sigma_sq.iloc[-3], sigma_sq.iloc[-3], sigma_sq.iloc[-2])
        self.params_ = pd.DataFrame({"link_ratio": link_ratio, "sigma_sq": sigma_sq, "volume": volume})
        return self

    def predict(self, new_df, process_scale=1.0):
        """
        Project the new book to ultimate and compute Mack standard errors and ULR percentiles.

        For origin i with latest age a_i and latest loss C_i::

            U_i = C_i * prod_{k >= a_i} f_k
            process variance   = U_i^2 * sum_k (scale * sigma_k^2 / f_k^2) / C_ik
            parameter variance = U_i^2 * sum_k (sigma_k^2 / f_k^2) / S_k

        where C_ik is the projected loss at future age k. Parameter variance stays in
        legacy units, because it depends on the legacy volumes S_k. The total adds
        Mack's cross term for the shared pattern::

            sum_{i<j} 2 U_i U_j sum_{k >= max(a_i, a_j)} (sigma_k^2 / f_k^2) / S_k

        Each ultimate (and the total) gets a moment-matched lognormal::

            ln_sigma = sqrt(ln(1 + CV^2)),  ln_mu = ln(U / premium) - ln_sigma^2 / 2

        Premium is treated as known, so ULR = U / premium is lognormal too.

        Parameters
        ----------
        new_df : pandas.DataFrame
            Long-format new-book data with loss and premium columns. Every age in it
            must also appear in the legacy pattern.
        process_scale : float, default 1.0
            Multiplier on sigma^2 for process variance, typically
            ``volume_ratio(new_df, legacy_df)``. Use 1.0 when ``new_df`` is the
            legacy book itself.

        Returns
        -------
        pandas.DataFrame
            One row per origin plus "Total", with columns:

            latest_age, latest_loss, premium : latest diagonal.
            ultimate : projected ultimate loss.
            process_se, parameter_se : SE components (NaN for Total).
            total_se : SE of the ultimate.
            ln_mu, ln_sigma : ULR lognormal parameters (log scale).
            ulr_mean : ultimate / premium.
            ulr_p05, ulr_p25, ulr_p50, ulr_p75, ulr_p95 : ULR percentiles.

        Raises
        ------
        ValueError
            If the loss and premium triangles differ.
        """
        params = self.params_
        latest_age, latest_loss, latest_premium = latest_loss_and_premium(
            new_df, self.origin_col, self.dev_col, self.loss_col, self.premium_col)

        rows = []
        for origin in latest_age.index:
            future = params[params.index >= latest_age[origin]]
            f, s2 = future["link_ratio"], future["sigma_sq"]
            projected_loss = latest_loss[origin] * f.shift(fill_value=1.0).cumprod()
            ultimate = latest_loss[origin] * f.prod()
            process_var = ultimate**2 * (process_scale * s2 / f**2 / projected_loss).sum()
            parameter_var = ultimate**2 * (s2 / f**2 / future["volume"]).sum()
            rows.append(dict(origin=origin, latest_age=latest_age[origin], latest_loss=latest_loss[origin],
                             premium=latest_premium[origin], ultimate=ultimate,
                             process_se=np.sqrt(process_var), parameter_se=np.sqrt(parameter_var)))
        origins = pd.DataFrame(rows).set_index("origin")
        origins["total_se"] = np.hypot(origins.process_se, origins.parameter_se)

        self.results_ = self._add_lognormal(pd.concat([origins, self._total(origins)]))
        return self.results_

    def ulr_distribution(self, origin):
        """
        Frozen scipy lognormal for one origin's (or the total's) ULR.

        Parameters
        ----------
        origin : int or str
            An origin in ``results_``, or "Total".

        Returns
        -------
        scipy.stats.rv_continuous_frozen
            With ``.pdf``, ``.cdf``, ``.ppf`` and ``.rvs``.
        """
        row = self.results_.loc[origin]
        return stats.lognorm(s=row.ln_sigma, scale=np.exp(row.ln_mu))

    def _total(self, origins):
        """Portfolio row: summed ultimates, SE including Mack's cross term."""
        params, cross_var, index = self.params_, 0.0, origins.index
        for i, older in enumerate(index):
            for younger in index[i + 1:]:
                p = params[params.index >= max(origins.at[older, "latest_age"], origins.at[younger, "latest_age"])]
                cross_var += 2 * origins.at[older, "ultimate"] * origins.at[younger, "ultimate"] * \
                             (p.sigma_sq / p.link_ratio**2 / p.volume).sum()
        return pd.DataFrame([dict(latest_loss=origins.latest_loss.sum(), premium=origins.premium.sum(),
                                  ultimate=origins.ultimate.sum(),
                                  total_se=np.sqrt((origins.total_se**2).sum() + cross_var))], index=["Total"])

    def _add_lognormal(self, result):
        """Add moment-matched lognormal parameters and ULR percentiles."""
        cv = result.total_se / result.ultimate
        result["ln_sigma"] = np.sqrt(np.log1p(cv**2))
        result["ln_mu"] = np.log(result.ultimate / result.premium) - result.ln_sigma**2 / 2
        result["ulr_mean"] = result.ultimate / result.premium
        for q in self.PERCENTILES:
            result[f"ulr_p{int(q * 100):02d}"] = np.exp(result.ln_mu + result.ln_sigma * stats.norm.ppf(q))
        return result
