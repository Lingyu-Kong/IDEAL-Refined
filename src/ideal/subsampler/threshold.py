from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from pydantic import BaseModel, NonNegativeFloat, NonNegativeInt, PositiveInt
from typing_extensions import override


class UncThresholdConfig(BaseModel):
    @abstractmethod
    def create_calculator(self) -> UncThresholdCalculator:
        pass


class UncThresholdCalculator(ABC):
    @abstractmethod
    def update(self, unc_value: float):
        pass

    @abstractmethod
    def get_threshold(self) -> float:
        pass

    @abstractmethod
    def reset(self):
        pass


class ValueUncThresholdConfig(UncThresholdConfig):
    """
    Configuration for uncertainty threshold calculator.
    """

    window_size: NonNegativeInt = 10000
    """
    window size for replay buffer
    """

    sigma: NonNegativeFloat
    """
    threshold = mean + sigma
    """

    window_size: PositiveInt = 10000

    @override
    def create_calculator(self):
        return ValueUncThresholdCalculator(
            window_size=self.window_size,
            sigma=self.sigma,
        )


class ValueUncThresholdCalculator(UncThresholdCalculator):
    def __init__(self, window_size, sigma):
        """
        Initialize the threshold calculator.

        Args:
            sigma (float): Scaling factor for std in mean+std threshold.
        """
        self.window_size = window_size
        self.sigma = sigma

        if self.window_size > 0:
            self.replay_buffer = deque(maxlen=self.window_size)
        else:
            self.replay_buffer = []

    @override
    def update(self, unc_value: float):
        self.replay_buffer.append(unc_value)

    @override
    def get_threshold(self):
        mean = np.mean(self.replay_buffer).item()
        threshold = mean + self.sigma
        return threshold

    @override
    def reset(self):
        self.replay_buffer.clear()


class PercentileUncThresholdConfig(UncThresholdConfig):
    """
    Configuration for uncertainty threshold calculator.
    """

    window_size: NonNegativeInt = 10000
    """
    window size for replay buffer
    """
    alpha: NonNegativeFloat = 0.5
    """
    alpha * count_percentile + (1 - alpha) * mean + k * std
    """
    beta: NonNegativeFloat = 0.9
    """
    moving average for mean and std
    """
    k: NonNegativeFloat = 1.0
    """
    mean + k * std
    """
    min_samples: PositiveInt = 100
    """
    minimum number of samples to use count_percentile
    less than this number, only use mean + k * std
    """

    def __init__(self, **data):
        super().__init__(**data)
        assert 0 <= self.alpha <= 1
        assert 0 <= self.beta < 1

    @override
    def create_calculator(self):
        return PercentileUncThresholdCalculator(
            window_size=self.window_size,
            alpha=self.alpha,
            beta=self.beta,
            k=self.k,
            min_samples=self.min_samples,
        )


class PercentileUncThresholdCalculator(UncThresholdCalculator):
    def __init__(self, window_size, alpha, beta, k, min_samples):
        """
        Initialize the threshold calculator.

        Args:
            window_size (int): Maximum size of the sliding window for replay buffer.
            alpha (float): Weighting factor for combining percentile and mean+std thresholds (0 <= alpha <= 1).
            beta (float): Weight for historical values in weighted mean and std updates (0 <= beta < 1).
            k (float): Scaling factor for std in mean+std threshold.
        """
        self.window_size = window_size
        self.alpha = alpha
        self.beta = beta
        self.k = k
        self.min_samples = min_samples

        if self.window_size > 0:
            self.replay_buffer = deque(maxlen=self.window_size)
        else:
            self.replay_buffer = []
        self.initialized = False

    @override
    def update(self, unc_value: float):
        """
        Update the replay buffer and statistics with new uncertainty values.

        Args:
            unc_values (list[float] | np.ndarray): New uncertainty values to add to the buffer.
        """
        self.replay_buffer.append(unc_value)

    @override
    def get_threshold(self, percentile=95):
        """
        Compute the combined threshold based on percentile and mean+std.

        Args:
            percentile (float): Percentile to calculate (0-100).

        Returns:
            float: The combined uncertainty threshold.
        """
        if len(self.replay_buffer) < self.min_samples:
            mean = 0.0
            std = 1.0
        else:
            mean = np.mean(self.replay_buffer).item()
            std = np.std(self.replay_buffer).item()
        mean_std_threshold = mean + self.k * std
        if len(self.replay_buffer) > self.min_samples:
            buffer_percentile = np.percentile(self.replay_buffer, percentile)
        else:
            buffer_percentile = mean_std_threshold

        # Combine thresholds using alpha
        combined_threshold = (
            self.alpha * buffer_percentile + (1 - self.alpha) * mean_std_threshold
        )
        print(
            f"mean: {mean}, std: {std}, percentile: {buffer_percentile}, threshold: {combined_threshold}, buffer size: {len(self.replay_buffer)}"
        )
        return combined_threshold

    @override
    def reset(self):
        """
        Reset the replay buffer and statistics to default values.
        """
        self.replay_buffer.clear()
        self.mean = 0
        self.std = 1.0
        self.initialized = False
