"""Validate both models against chainladder on the genins sample (legacy book applied to itself)."""
import numpy as np
import pytest

cl = pytest.importorskip("chainladder")
from pattern_ulr import MackChainladder, BootChainladder


@pytest.fixture(scope="module")
def genins():
    triangle = cl.load_sample("genins")
    df = (triangle.to_frame(keepdims=True).reset_index()
          .rename(columns={"development": "dev", "values": "cumulative_loss"}))
    df["origin"] = df["origin"].dt.year
    df["earned_premium"] = 1.0
    return triangle, df


def test_mack_matches_chainladder(genins):
    triangle, df = genins
    ours = MackChainladder().fit(df).predict(df)
    ref = cl.MackChainladder().fit(cl.Development(sigma_interpolation="mack").fit_transform(triangle))
    ref_se = ref.summary_.to_frame(origin_as_datetime=False)["Mack Std Err"].to_numpy()
    np.testing.assert_allclose(ours.total_se.iloc[1:-1], ref_se[1:], rtol=1e-9)        # first origin fully developed
    np.testing.assert_allclose(ours.at["Total", "total_se"], float(ref.total_mack_std_err_.values.ravel()[0]), rtol=1e-9)


def test_bootstrap_matches_chainladder(genins):
    triangle, df = genins
    model = BootChainladder(n_sims=10_000, seed=0).fit(df)
    model.predict(df)
    reserve = model.ultimates_["Total"] - df.groupby("origin").cumulative_loss.max().sum()
    ref = cl.Chainladder().fit(cl.BootstrapODPSample(n_sims=10_000, random_state=0, hat_adj=False)
                               .fit_transform(triangle)).ibnr_.sum("origin").values.ravel()
    assert reserve.mean() == pytest.approx(ref.mean(), rel=0.02)
    assert reserve.std() == pytest.approx(ref.std(), rel=0.05)
