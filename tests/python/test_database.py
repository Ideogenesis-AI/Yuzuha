# Copyright (C) 2026 Changkai Zhang.
#
# This file is part of Yuzuha library.
#
# Yuzuha is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published
# by the Free Software Foundation, either version 3 of the License,
# or (at your option) any later version.
#
# Yuzuha is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Yuzuha. If not, see <https://www.gnu.org/licenses/>.

"""
Pre-computation of the CG basis database.

`startup_database` sweeps a grid of spin configurations and warms the cache.
Its default grid takes a long time, so these tests pass narrow `special_spins`
and `general_spins` ranges; the control flow being exercised is the same.
"""

import logging

import pytest

import yuzuha
from yuzuha.database import (
    DEFAULT_GENERAL_SPINS,
    DEFAULT_SPECIAL_SPINS,
    startup_database,
)


@pytest.fixture
def records(caplog):
    """Capture everything startup_database logs through the default logger."""
    caplog.set_level(logging.DEBUG, logger="yuzuha.database")
    return caplog


def messages(caplog):
    return [r.getMessage() for r in caplog.records]


class TestDefaults:
    def test_default_ranges_are_unchanged(self):
        # The defaults are the documented j=1/2, 1 and j=1/2 ... 3 sweeps.
        assert DEFAULT_SPECIAL_SPINS == (1, 2)
        assert DEFAULT_GENERAL_SPINS == (1, 2, 3, 4, 5, 6)

    def test_is_exported(self):
        assert yuzuha.startup_database is startup_database

    def test_omitted_ranges_fall_back_to_the_defaults(self, monkeypatch, caplog):
        # The real defaults sweep far too much work for a unit test, so the
        # constants themselves are narrowed; what matters is that omitting the
        # arguments reads them instead of using some other grid.
        monkeypatch.setattr(yuzuha.database, "DEFAULT_SPECIAL_SPINS", (1,))
        monkeypatch.setattr(yuzuha.database, "DEFAULT_GENERAL_SPINS", (1, 2))
        caplog.set_level(logging.DEBUG, logger="yuzuha.database")

        startup_database()

        logged = [r.getMessage() for r in caplog.records]
        assert any("one spin ∈ {1/2}" in m for m in logged)
        assert any("two spins ∈ {1/2, 1}" in m for m in logged)


class TestLogging:
    def test_creates_a_default_logger(self, records):
        startup_database(special_spins=[1], general_spins=[1, 2])

        logged = messages(records)
        assert any("Starting CG basis database pre-computation" in m for m in logged)
        assert any("Computing 3rd order bases" in m for m in logged)
        assert any("Computing 4th order bases" in m for m in logged)
        assert any("pre-computation complete" in m for m in logged)

        # The default logger must be configured with exactly one handler, and
        # a second call must not stack another one on top.
        logger = logging.getLogger("yuzuha.database")
        before = len(logger.handlers)
        startup_database(special_spins=[1], general_spins=[1])
        assert len(logger.handlers) == before

    def test_accepts_a_caller_supplied_logger(self, caplog):
        logger = logging.getLogger("test_startup_database.custom")
        caplog.set_level(logging.DEBUG, logger=logger.name)

        startup_database(logger=logger, special_spins=[1], general_spins=[1, 2])

        assert any(r.name == logger.name for r in caplog.records)

    def test_log_level_applies_to_the_default_logger(self, records):
        startup_database(log_level=logging.WARNING, special_spins=[1], general_spins=[1])
        assert logging.getLogger("yuzuha.database").level == logging.WARNING


class TestSweep:
    def test_reports_progress_and_a_summary(self, records):
        # A three-value general range yields well over ten third-order
        # configurations, so both the "every tenth" and the "final index"
        # progress branches fire.
        startup_database(special_spins=[1], general_spins=[1, 2, 3])

        logged = messages(records)
        assert any("Progress:" in m for m in logged)
        assert any("Total configurations attempted:" in m for m in logged)
        assert any("Successfully computed:" in m for m in logged)
        assert any("Failed:" in m for m in logged)

    def test_skips_configurations_that_cannot_couple_to_zero(self, records):
        # Mixing half-integer and integer spins produces combinations with an
        # odd number of half-integers, which have no valid fusion tree. Those
        # are expected and must be logged at debug level, not as errors, and
        # must not count towards the attempted total.
        startup_database(log_level=logging.DEBUG, special_spins=[1], general_spins=[1, 2])

        logged = messages(records)
        assert any("Skipped invalid configuration" in m for m in logged)
        assert not any(r.levelno >= logging.ERROR for r in records.records)

        attempted = next(m for m in logged if "Total configurations attempted:" in m)
        enumerated = sum(
            int(m.rsplit(":", 1)[1]) for m in logged if "Total 3rd order" in m or "Total 4th order" in m
        )
        assert int(attempted.rsplit(":", 1)[1]) < enumerated

    def test_unexpected_failures_are_logged_as_errors(self, monkeypatch, records):
        def boom(_spec):
            raise RuntimeError("synthetic backend failure")

        monkeypatch.setattr(yuzuha.database, "canonical_basis", boom)
        startup_database(special_spins=[1], general_spins=[1, 2])

        errors = [r.getMessage() for r in records.records if r.levelno >= logging.ERROR]
        assert errors, "an unexpected exception must be reported"
        assert any("Failed for configuration" in m for m in errors)
        assert any("synthetic backend failure" in m for m in errors)

    def test_single_spin_sweep_populates_the_cache(self, records):
        # j=1/2 everywhere is the smallest sweep that still builds real bases.
        startup_database(special_spins=[1], general_spins=[1])

        summary = [m for m in messages(records) if "Successfully computed:" in m]
        assert len(summary) == 1
        count = int(summary[0].rsplit(":", 1)[1])
        assert count > 0, "at least one configuration must couple to j=0"
