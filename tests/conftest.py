from __future__ import annotations

import pytest

import paguro as pg

pg.Config.set_styled(styled=False)  # TODO: look into styling error when running pytest


@pytest.fixture
def pizza_menu():
    return pg.dataset.pizza("menu")


@pytest.fixture
def pizza_orders():
    return pg.dataset.pizza("orders")
