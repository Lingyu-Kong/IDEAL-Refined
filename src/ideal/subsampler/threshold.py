from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from pydantic import BaseModel, NonNegativeFloat, PositiveInt
from typing_extensions import override


class UncThresholdConfig(BaseModel):
    @abstractmethod
    def create_calculator(self) -> UncThresholdCalculator:
        pass


class UncThresholdCalculator(ABC):
    @abstractmethod
    def update(self, unc_values):
        pass

    @abstractmethod
    def get_threshold(self, percentile=95):
        pass

    @abstractmethod
    def reset(self):
        pass


class PercentileUncThresholdConfig(UncThresholdConfig):
    """
    Configuration for uncertainty threshold calculator.
    """

    window_size: PositiveInt = 1000
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

        self.replay_buffer = deque(maxlen=self.window_size)
        self.weighted_mean = 0
        self.weighted_std = 1.0
        self.initialized = False

    def _update_stats(self, new_value):
        """
        Update weighted mean and std using the new value and historical stats.
        """
        if not self.initialized:
            # Initialize with the first value
            self.weighted_mean = new_value
            self.weighted_std = 0
            self.initialized = True
        else:
            # Update weighted mean
            self.weighted_mean = (
                self.beta * self.weighted_mean + (1 - self.beta) * new_value
            )
            # Update weighted std
            self.weighted_std = np.sqrt(
                self.beta * (self.weighted_std**2)
                + (1 - self.beta) * ((new_value - self.weighted_mean) ** 2)
            )

    @override
    def update(self, unc_values):
        """
        Update the replay buffer and statistics with new uncertainty values.

        Args:
            unc_values (list[float] | np.ndarray): New uncertainty values to add to the buffer.
        """
        unc_values = np.array(unc_values)
        for value in unc_values:
            self.replay_buffer.append(value)
            self._update_stats(value)

    @override
    def get_threshold(self, percentile=95):
        """
        Compute the combined threshold based on percentile and mean+std.

        Args:
            percentile (float): Percentile to calculate (0-100).

        Returns:
            float: The combined uncertainty threshold.
        """
        mean_std_threshold = self.weighted_mean + self.k * self.weighted_std
        if len(self.replay_buffer) > self.min_samples:
            buffer_percentile = np.percentile(self.replay_buffer, percentile)
        else:
            buffer_percentile = mean_std_threshold

        # Combine thresholds using alpha
        combined_threshold = (
            self.alpha * buffer_percentile + (1 - self.alpha) * mean_std_threshold
        )
        return combined_threshold

    @override
    def reset(self):
        """
        Reset the replay buffer and statistics to default values.
        """
        self.replay_buffer.clear()
        self.weighted_mean = 0
        self.weighted_std = 1.0
        self.initialized = False
