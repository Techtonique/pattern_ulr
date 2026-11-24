"""
Shared helpers: triangle construction, latest diagonal, book-size scaling.

Input data
----------
One long DataFrame per book, one row per (origin, development age)::

    origin | dev | cumulative_loss | earned_premium
    -------+-----+-----------------+---------------
    2020   |  12 |            1000 |           5000
    2020   |  24 |            1800 |           5000

origin : int or label
    Accident or underwriting period.
dev : int
    Development age (e.g. months). Must sort numerically (not strings) and be on
    the same grid in both books. Spacing need not be regular.
cumulative_loss : float
    Cumulative paid or incurred loss.
earned_premium : float
    Cumulative earned premium. Needed for the new book; optional for the legacy
    book (it lets you compute legacy ULRs and size the books by premium).

Losses and premium share the (origin, dev) keys, so they are two triangles held in
one table and no join is needed. Column names are set on each model's constructor.
"""
import pandas as pd


def to_triangle(long_df, origin_col, dev_col, value_col):
    """
    Pivot a long table into a wide triangle.

    Parameters
    ----------
    long_df : pandas.DataFrame
        Long-format data, one row per (origin, development age).
    origin_col : str
        Column holding the origin period.
    dev_col : str
        Column holding the development age.
    value_col : str
        Column holding the values to place in the triangle.

    Returns
    -------
    pandas.DataFrame
        Origins as the index and development ages as the columns, both sorted
        ascending. Cells not yet observed are NaN.
    """
    return long_df.pivot(index=origin_col, columns=dev_col, values=value_col).sort_index().sort_index(axis=1)


def latest_diagonal(triangle):
    """
    Find each origin's latest observed age and its value.

    Parameters
    ----------
    triangle : pandas.DataFrame
        Wide triangle from ``to_triangle``.

    Returns
    -------
    latest_age : pandas.Series
        Latest development age with a non-NaN value, indexed by origin.
    latest_value : pandas.Series
        The value at that age, indexed by origin.
    """
    latest_age = triangle.apply(lambda row: row.last_valid_index(), axis=1)
    latest_value = pd.Series({o: triangle.at[o, a] for o, a in latest_age.items()})
    return latest_age, latest_value


def latest_loss_and_premium(new_df, origin_col, dev_col, loss_col, premium_col):
    """
    Build the loss and premium triangles of the new book and read their latest diagonal.

    Parameters
    ----------
    new_df : pandas.DataFrame
        Long-format data for the new book, with both loss and premium columns.
    origin_col, dev_col, loss_col, premium_col : str
        Column names in ``new_df``.

    Returns
    -------
    latest_age : pandas.Series
        Latest observed age per origin.
    latest_loss : pandas.Series
        Cumulative loss at that age.
    latest_premium : pandas.Series
        Earned premium at that age (the ULR denominator).

    Raises
    ------
    ValueError
        If the loss and premium triangles differ in origins, ages or observed cells.
    """
    loss_triangle = to_triangle(new_df, origin_col, dev_col, loss_col)
    premium_triangle = to_triangle(new_df, origin_col, dev_col, premium_col)
    if not loss_triangle.notna().equals(premium_triangle.notna()):
        raise ValueError("Loss and premium triangles must share the same origins, ages and observed cells.")
    latest_age, latest_loss = latest_diagonal(loss_triangle)
    _, latest_premium = latest_diagonal(premium_triangle)
    return latest_age, latest_loss, latest_premium


def volume_ratio(new_df, legacy_df, dev_col="dev", size_col="cumulative_loss"):
    """
    Size of the new book relative to the legacy book.

    Process variance (Mack's sigma^2, the ODP scale phi) grows with book size, so
    legacy values must be rescaled before being applied to the new book. This
    measures size as the mean of ``size_col`` at the new book's first development
    age: cumulative loss by default, or earned premium (``size_col="earned_premium"``)
    when both books have it, which is usually the more stable measure.

    Parameters
    ----------
    new_df : pandas.DataFrame
        Long-format data for the new (immature) book.
    legacy_df : pandas.DataFrame
        Long-format data for the legacy (mature) book.
    dev_col : str, default "dev"
        Column holding the development age.
    size_col : str, default "cumulative_loss"
        Column measuring book size: cumulative loss, or earned premium if both books have it.

    Returns
    -------
    float
        Mean new-book value divided by mean legacy value, at the first age. Pass it
        as ``process_scale`` to ``predict``.
    """
    first_age = new_df[dev_col].min()
    mean_at_first_age = lambda df: df.loc[df[dev_col] == first_age, size_col].mean()
    return mean_at_first_age(new_df) / mean_at_first_age(legacy_df)