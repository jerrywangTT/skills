# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
"""pytest plugin the push queue and `ttp checks` load (`-p ttp_pytest_durations`, slowtests.env) to
keep a check's slowest tests.

TTP_PYTEST_DURATIONS names the JSON file to write. A test's time is the sum of its setup, call and
teardown durations, as pytest's --durations counts them, from the reports the main process sees
(xdist or forked workers send theirs there). At the session's end the TTP_PYTEST_DURATIONS_TOP
slowest (default 30) are written to the file, merged with what an earlier pytest run of the same
check left in it: {"tests": [{"id", "s"}], "count": tests timed}. It selects, skips and changes
nothing else. It acts only in the pytest process it was loaded into: it takes its load and file out
of the environment, so a pytest a test starts runs without it."""
from __future__ import annotations

import json
import os

ENV, TOP_ENV, NAME, TOP = "TTP_PYTEST_DURATIONS", "TTP_PYTEST_DURATIONS_TOP", "ttp_pytest_durations", 30


def _strip_env() -> tuple[str, str]:
    """(file, top), with the plugin's load and settings taken out of the environment children inherit."""
    path, top = os.environ.pop(ENV, ""), os.environ.pop(TOP_ENV, "")
    words = os.environ.get("PYTEST_ADDOPTS", "").split()
    kept = [w for i, w in enumerate(words) if not (w == "-p" and words[i + 1:i + 2] == [NAME]
                                                   or w == NAME and words[i - 1:i] == ["-p"])]
    if kept:
        os.environ["PYTEST_ADDOPTS"] = " ".join(kept)
    else:
        os.environ.pop("PYTEST_ADDOPTS", None)
    return path, top


class Durations:
    def __init__(self, path: str, top: str):
        self.path, self.times = path, {}
        try:
            self.top = max(int(top or TOP), 1)
        except ValueError:
            self.top = TOP

    def pytest_runtest_logreport(self, report):
        self.times[report.nodeid] = self.times.get(report.nodeid, 0.0) + float(getattr(report, "duration", 0) or 0)

    def pytest_sessionfinish(self, session, exitstatus):
        if not self.times:
            return
        times, count = dict(self.times), len(self.times)
        try:
            with open(self.path) as f:
                old = json.load(f)
            times = {str(t["id"]): float(t["s"]) for t in old.get("tests", [])} | times
            count += int(old.get("count", 0))
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            pass
        rows = sorted(times.items(), key=lambda kv: -kv[1])[:self.top]
        tmp = f"{self.path}.{os.getpid()}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"tests": [{"id": k, "s": round(v, 3)} for k, v in rows], "count": count}, f)
            os.replace(tmp, self.path)
        except OSError:   # a timing note never fails the run
            try:
                os.unlink(tmp)
            except OSError:
                pass


def pytest_configure(config):
    path, top = _strip_env()
    if path and not hasattr(config, "workerinput"):   # an xdist worker's reports reach the controller
        config.pluginmanager.register(Durations(path, top), "ttp-pytest-durations")
