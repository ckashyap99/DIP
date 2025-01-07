"""The assignment for Data-Intensive Programming 2022"""

from typing import List, Tuple

from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


class Assignment:
    spark: SparkSession = None  # REPLACE with actual implementation

    # the data frame to be used in tasks 1 and 4
    dataD2: DataFrame = None  # REPLACE with actual implementation

    # the data frame to be used in task 2
    dataD3: DataFrame = None  # REPLACE with actual implementation

    # the data frame to be used in task 3 (based on dataD2 but containing numeric labels)
    dataD2WithLabels: DataFrame = None  # REPLACE with actual implementation



    @staticmethod
    def task1(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        pass  # REPLACE with actual implementation

    @staticmethod
    def task2(df: DataFrame, k: int) -> List[Tuple[float, float, float]]:
        pass  # REPLACE with actual implementation

    @staticmethod
    def task3(df: DataFrame, k: int) -> List[Tuple[float, float]]:
        pass  # REPLACE with actual implementation

    # Parameter low is the lowest k and high is the highest one.
    @staticmethod
    def task4(df: DataFrame, low: int, high: int) -> List[Tuple[int, float]]:
        pass  # REPLACE with actual implementation
