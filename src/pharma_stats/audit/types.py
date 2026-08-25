"""Shared result type for every audit check.

Every check states a numeric expectation and an observed actual value —
a check that can't fail is decoration, not a check.
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("FAIL", "WARN", "INFO", "PASS")


@dataclass
class Check:
    stage: str
    name: str
    level: str  # one of LEVELS
    expected: str
    actual: str
    detail: str = ""


def fail(stage: str, name: str, expected: str, actual: str, detail: str = "") -> Check:
    return Check(stage, name, "FAIL", expected, actual, detail)


def warn(stage: str, name: str, expected: str, actual: str, detail: str = "") -> Check:
    return Check(stage, name, "WARN", expected, actual, detail)


def info(stage: str, name: str, expected: str, actual: str, detail: str = "") -> Check:
    return Check(stage, name, "INFO", expected, actual, detail)


def ok(stage: str, name: str, expected: str, actual: str, detail: str = "") -> Check:
    return Check(stage, name, "PASS", expected, actual, detail)
