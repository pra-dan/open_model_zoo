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

from collections import namedtuple
import numpy as np

from ..config import NumberField, PathField
from ..representation import ClassificationAnnotation
from ..utils import read_txt, get_path, check_file_existence

from .format_converter import BaseFormatConverter, ConverterReturn

AudioInfo = namedtuple('AudioInfo', ['file', 'fold', 'target', 'category', 'esc10', 'src_file', 'take'])


class EscFormatConverter(BaseFormatConverter):
    __provider__ = 'esc'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'annotation_file': PathField(description="Path to annotation in cvs format."),
            'fold': NumberField(
                optional=True, default=-1,
                description="Set fold for validation"),
            'audio_dir': PathField(
                is_directory=True, optional=True,
                description='Path to dataset audio files, used only for content existence check'
            )
        })

        return parameters

    def configure(self):
        self.annotation_file = self.get_value_from_config('annotation_file')
        self.audio_dir = self.get_value_from_config('audio_dir') or self.annotation_file.parent
        self.fold = self.get_value_from_config('fold')

    def convert(self, check_content=False, progress_callback=None, progress_interval=100, **kwargs):
        annotation = []
        content_errors = [] if check_content else None
        original_annotation = read_txt(get_path(self.annotation_file))
        num_iterations = len(original_annotation)
        for audio_id, audio in enumerate(original_annotation):
            info = AudioInfo(audio.split(','))
            if int(info.fold) != self.fold and self.fold != -1:
                continue
            if check_content:
                if not check_file_existence(self.audio_dir / info.file):
                    content_errors.append('{}: does not exist'.format(self.audio_dir / info.file))

            annotation.append(ClassificationAnnotation(info.file, np.int64(info.target)))
            if progress_callback is not None and audio_id % progress_interval == 0:
                progress_callback(audio_id / num_iterations * 100)

        return ConverterReturn(annotation, None, content_errors)
