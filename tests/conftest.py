"""Fixtures: the toy canonical set of :mod:`lrbench.toy`, whose "DFT" references are a Lennard-Jones calculator's own
values, so a model that *is* that calculator must score zero error and tail fraction 1."""
import pytest

from lrbench.toy import LJ_SPEC as LJ  # noqa: F401  (re-exported for tests)
from lrbench.toy import build, lj_calc


def lj():
    return lj_calc()


@pytest.fixture(scope="session")
def datadir(tmp_path_factory):
    return build(str(tmp_path_factory.mktemp("data")))
