#
# Copyright (c) 2020, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Logger created with Piotr Januszewski: https://piojanu.github.io/

import collections
import re

import acme.utils.loggers as acme_loggers
import acme.utils.tree_utils as acme_tree
import numpy as np


class NeptuneLogger(acme_loggers.base.Logger):
    """Neptune logger for Acme.

    Args:
        prefix (string): The string used to prefix data keys in a name of a log.
          Can be None in which case no prefix is used.
        index_name (string): The data key which value to use as a log index.
          Can be None in which case no index is used.
    """

    def __init__(self, experiment, prefix=None, index_name=None):
        super()
        self._experiment = experiment
        self._prefix = prefix
        self._index_name = index_name or ''

    def write(self, values):
        """Send `values` to Neptune."""
        values = acme_loggers.to_numpy(values)
        index = values.pop(self._index_name, None)
        for key, value in values.items():
            prefixed_key = f'{self._prefix}/{key}' if self._prefix else key
            if index:
                self._experiment.log_metric(prefixed_key, index, value)
            else:
                self._experiment.log_metric(prefixed_key, value)


class AggregateFilter(acme_loggers.base.Logger):
    """Logger which writes to another logger, aggregating matching data.

    Args:
        to (Logger): An object to which the current object will forward the
          aggregated data when `dump` is called.
        aggregate_regex (string): A regex of data keys which should be
          aggregated.

    Note:
        For not matched keys the last value will be forwarded.
    """

    def __init__(self, to, aggregate_regex):
        super()
        self._to = to
        self._aggregate_regex = aggregate_regex

        self._cache = []

    def write(self, values):
        self._cache.append(values)

    def dump(self):
        """Calculates statistics and forwards them to the target logger."""
        results = {}

        stacked_cache = acme_tree.stack_sequence_fields(self._cache)
        for key, values in stacked_cache.items():
            if re.search(self._aggregate_regex, key) is not None:
                results.update({
                    f'{key}_mean': np.mean(values),
                    f'{key}_std': np.std(values),
                    f'{key}_median': np.median(values),
                    f'{key}_max': np.max(values),
                    f'{key}_min': np.min(values),
                })
            else:
                results[key] = values[-1]

        self._to.write(results)
        self._cache.clear()


class SmoothingFilter(acme_loggers.base.Logger):
    """Logger which writes to another logger, smoothing matching data.

    Args:
        to (Logger): An object to which the current object will forward the
          original data and its results when `write` is called.
        smoothing_regex (string): A regex of data keys which should be smoothed.
        smoothing_coeff (float): A desired smoothing strength between 0 and 1.

    Note:
        For example values of regex = 'return' and coeff = 0.99 will calculate
        the running average of all data which contain 'return' in their key.
        It's calculated according to: average = 0.99 * average + 0.01 * value.
        Warm-up period of length 10 is also applied (see the comment in code).
    """

    def __init__(self, to, smoothing_regex, smoothing_coeff):
        super()
        self._to = to
        self._smoothing_regex = smoothing_regex
        self._smoothing_coeff = smoothing_coeff

        self._previous_values = collections.defaultdict(float)
        self._smoothing_coeffs = collections.defaultdict(float)

    def write(self, values):
        """Smooths matching data and forwards it with the original data."""
        values_ = dict(values)

        for key, value in values.items():
            if re.search(self._smoothing_regex, key) is not None:
                smoothed_key = f'{key}_smoothed_{self._smoothing_coeff}'
                prev_value = self._previous_values[smoothed_key]
                prev_smoothing_coeff = self._smoothing_coeffs[smoothed_key]

                # This implements warm-up period of length "10". That is
                # the smoothing coefficient start with 0 and is annealed to
                # the desired value.
                # This results in better estimates of smoothed value at the
                # beginning, which might be useful for short experiments.
                new_smoothing_coeff = (prev_smoothing_coeff * 0.9 +
                                       self._smoothing_coeff * 0.1)
                smoothed_value = (value * (1 - prev_smoothing_coeff) +
                                  prev_value * prev_smoothing_coeff)

                self._previous_values[smoothed_key] = smoothed_value
                self._smoothing_coeffs[smoothed_key] = new_smoothing_coeff
                values_[smoothed_key] = smoothed_value

        self._to.write(values_)
