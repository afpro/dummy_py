# Copyright 2018, afpro <admin@afpro.net>.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#    http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================
import librosa
import numpy as np


def mfcc(y: 'np.ndarray',
         sr: 'int',
         n_mfcc: 'int' = 20,
         n_stft: 'int' = 256,
         hop_length: 'int' = None,
         freq_min: 'float' = 0.0,
         freq_max: 'float' = None,
         center: 'bool' = None):
    """
    :param y: input pcm
    :param sr: sample rate
    :param n_mfcc: mfcc level count
    :param n_stft: stft frame length
    :param hop_length: number audio of frames between stft columns
    :param freq_min: lowest frequency (in Hz)
    :param freq_max: highest frequency (in Hz).
    :param center: if `True`, the signal `y` is padded
    :return: mfcc data, as shape (n_mfcc, mfcc_count)
    """
    if hop_length is None:
        hop_length = n_stft
    if center is None:
        center = len(y) < n_stft
    fft = librosa.core.stft(y=y, n_fft=n_stft, hop_length=hop_length, center=center)
    db = librosa.power_to_db(np.abs(fft) ** 2)
    mel = np.dot(librosa.filters.mel(sr=sr, n_fft=n_stft, fmin=freq_min, fmax=freq_max, n_mels=n_mfcc), db)
    return np.dot(librosa.filters.dct(n_mfcc, mel.shape[0]), mel)
