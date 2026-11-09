"""Shared test helpers. Report lines added through the `report` fixture are printed in a
"fsbench report" section at the end of the pytest run (visible without -s)."""

import pytest

_REPORT = []


@pytest.fixture
def report():
    return _REPORT.append


def pytest_terminal_summary(terminalreporter):
    if _REPORT:
        terminalreporter.section("fsbench report")
        for line in _REPORT:
            terminalreporter.write_line(line)
