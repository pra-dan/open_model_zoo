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


class ResampleAudio(Preprocessor):
    __provider__ = 'resample_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'sample_rate': NumberField(value_type=int, min_value=1,
                                       description='Set new audio sample rate.'),
        })
        return parameters

    def configure(self):
        self.sample_rate = self.get_value_from_config('sample_rate')

    def process(self, image, annotation_meta=None):
        try:
            sample_rate = annotation_meta['sample_rate']
        except KeyError:
            raise RuntimeError('Operation "{}" can\'t resample audio: required original samplerate in metadata.'.
                               format(self.__provider__))
        except AttributeError:
            raise RuntimeError('Operation "{}" can\'t resample audio: required annotation metadata.'.
                               format(self.__provider__))

        if sample_rate == self.sample_rate:
            return image

        data = image.data
        duration = data.shape[1] / sample_rate
        resampled_data = np.zeros(shape=(1, duration*self.sample_rate), dtype=float)
        x_old = np.linspace(0, duration, data.shape[1])
        x_new = np.linspace(0, duration, resampled_data.shape[1])
        resampled_data = np.interp(x_new, x_old, data)

        image.data = resampled_data
        image.metadata['sample_rate'] = self.sample_rate

        return image


class ClipAudio(Preprocessor):
    __provider__ = 'clip_audio'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, min_value=0,
                description="Length of audio clip in milliseconds."
            ),
            'max_clips': NumberField(
                value_type=int, min_value=1, description="Maximum number of clips per audiofile."
            ),
        })
        return parameters

    def configure(self):
        self.size = self.get_value_from_config('size')
        self.max_clips = self.get_value_from_config('max_clips')

    def process(self, image, annotation_meta=None):
        data = image.data
        audio_size = data.shape[1]
        clipped_data = []
        for i in range(self.max_clips):
            if (i + 1)*self.size <= audio_size:
                clip = data[:, i * self.size: (i+1) * self.size]
            else:
                break
            clipped_data.append(clip)
        image.data = clipped_data
        image.metadata['multi_infer'] = True

        return image
