# -*- coding: utf-8 -*-
from .rboost_python import WordCounter, count_line

__all__ = ["RBoostRegressor", "fit"]


def search_py(path, needle):
    total = 0
    with open(path, "r") as f:
        for line in f:
            words = line.split(" ")
            for word in words:
                if word == needle:
                    total += 1
    return total