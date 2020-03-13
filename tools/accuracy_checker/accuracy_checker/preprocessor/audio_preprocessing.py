"""
Copyright (c) 2020 Intel Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np

from ..config import ConfigError, BaseField, NumberField, ListField, StringField
from ..preprocessor import Preprocessor
from ..utils import get_or_parse_value


class AudioResample(Preprocessor):
    __provider__ = 'audio_resample'

    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'sample_rate': NumberField(value_type=int, min_value=1,
                                       description='Set new audio sample rate.'),
        })
        return parameters

    def configure(self):
        self.sample_rate = set.get_value_from_config('sample_rate')

    def process(self, image, annotation_meta=None):
        #try
        sample_rate = annotation_meta.get('sample_rate')
        #raise
        if sample_rate == self.sample_rate:
            return image

        data = image.data
        duration = data.shape[1] / sample_rate
        resampled_data = np.zeros(shape=(1, duration*self.sample_rate), dtype=float)
        x_old = np.linspace(0, duration, data.shape[1])
        x_new = np.linspace(0, duration, resampled_data.shape[1])
        resampled_data = np.interp(x_new, x_old, data)

        image.data = resampled_data

        return image
