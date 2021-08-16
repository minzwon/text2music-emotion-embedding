import math
import random
import librosa
import numpy as np
import torch
import torch.nn as nn
import torchaudio.functional as F
import torchaudio
from torchaudio.compliance import kaldi
from nnAudio.Spectrogram import CQT1992v2


###############################
# Data augmentation for audio #
###############################
class Gain(nn.Module):
    """
    Randomly apply gain.
    It applies a single value when db_min is equal to db_max.

    Args:
        db_min (float): minimum dB (Default: 0.0)
        db_max (float): maximum dB (Default: 0.0)

    Input:
        x (torch.tensor): input audio batch [batch, length]
    Returns:
        out (torch.tensor): audio with applied gain [batch, length]
    """
    def __init__(self,
                 db_min=0.0,
                 db_max=0.0):
        super(Gain, self).__init__()
        self.db_min = db_min
        self.db_max = db_max

    def forward(self, x):
        is_random = False if self.db_min==self.db_max else True
        if is_random:
            db = torch.zeros(x.size(0))
            gain = (10 ** (db.uniform_(self.db_min, self.db_max) / 20)).unsqueeze(1).to(x.device)
            return gain * x
        else:
            gain = 10 ** (self.db_min / 20)
            return gain * x


class GaussianNoise(nn.Module):
    """
    Add gaussian noise.
    Signal to noise ratio (SNR) will be randomized. But it applies a single value when snr_min is equal to snr_max.

    Reference:
            https://github.com/sleekEagle/audio_processing/blob/master/mix_noise.py

    Args:
        snr_min (float): minimum SNR (Default: 80)
        snr_max (float): maximum SNR (Default: 80)

    Input:
        x (torch.tensor): input audio batch [batch, length]
    Returns:
        out (torch.tensor): audio with noise [batch, length]
    """
    def __init__(self,
                 snr_min=80.0,
                 snr_max=80.0):
        super(GaussianNoise, self).__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, x):
        # gaussian noise with zero mean, unit variance
        noise = torch.zeros(x.size())

        # get SNR
        is_random = False if self.snr_min==self.snr_max else True
        if is_random:
            snr = torch.zeros(x.size(0))
            snr = snr.uniform_(self.snr_min, self.snr_max).to(x.device).unsqueeze(1)
        else:
            snr = self.snr_min * torch.ones(x.size(0)).unsqueeze(1)
                        
        # add noise
        rms_signal = torch.sqrt(torch.mean(x ** 2, dim=1))
        rms_noise = torch.sqrt(rms_signal ** 2 / pow(10, snr / 10))[0][0] # For gaussian noise, rms = std.
        noise = noise.normal_(0, rms_noise).to(x.device)
        out = noise + x
        return out

    
class STFT(nn.Module):
    """
    Short time fourier transform. Just made a Pytorch method to be a module.

    Args:
        n_fft (int): n_fft (Default: 512)
        hop_length (int): hop size (Default: 256)
        win_length (int): window length (Default: None)
        window (tensor): window tensor (Default: None)

    Input:
        x (torch.tensor): input audio batch [batch, length]
    Returns:
        stft (torch.tensor): output spectrogram with complex values[batch, n_bins, length, 2]
    """
    def __init__(self,
                 n_fft=512,
                 hop_length=256,
                 win_length=512,
                 window_fn=torch.hann_window):
        super(STFT, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window_fn(win_length)
        self.stft = torchaudio.transforms.Spectrogram(n_fft=n_fft, win_length=win_length, hop_length=hop_length,power=None)

    def forward(self, x):
        out = self.stft(x)
        return out


class ComplexNorm(torchaudio.transforms.ComplexNorm):
    """
    Args:
        power (float, optional): Power of the norm. (Default: to ``1.0``)
    Input:
        complex_spectrograms(torch.tensor): complex spectrogram (..., freq, time, complex=2)
    Returns:
        Tensor: norm of the input tensor.
    """
    pass


class MelScale(torchaudio.transforms.MelScale):
    """
    Args:
        n_mels (int, optional): Number of mel filterbanks. (Default: ``128``)
        sample_rate (int, optional): Sample rate of audio signal. (Default: ``16000``)
        f_min (float, optional): Minimum frequency. (Default: ``0.``)
        f_max (float or None, optional): Maximum frequency. (Default: ``sample_rate // 2``)
        n_stft (int, optional): Number of bins in STFT. Calculated from first input
                                if None is given.  See ``n_fft`` in :class:`Spectrogram`. (Default: ``None``)
    """
    pass


class AmplitudeToDB(torchaudio.transforms.AmplitudeToDB):
    pass


class CQT(nn.Module):
    """
    CQT using nnAudio.

    Reference:
            https://github.com/KinWaiCheuk/nnAudio

    Args:
        sample_rate (int): sample rate (Default: 16000)
        hop_length (int): hop length (Default: 256)
        fmin (float): minimum frequency of CQT bin. If None, it's C2 (Default: None)
        fmax (float): maximum frequency of CQT bin. If None, it's a half of the sample rate (Default: None)
        n_bins (int): number of CQT bins (Default: 128)

    Input:
        x (torch.tensor): input audio batch [batch, length]

    Returns:
        cqt (torch.tensor): CQT [batch, freq, time]
    """
    def __init__(self,
                 sample_rate=16000,
                 hop_length=256,
                 fmin=None,
                 fmax=None,
                 n_bins=128):
        super(CQT, self).__init__()
        if fmin is None:
            fmin = 2 * 32.703195		# note C2
        if fmax is None:
            fmax = sample_rate / 2
        bins_per_octave = n_bins / ((librosa.core.hz_to_midi(fmax) - librosa.core.hz_to_midi(fmin))/12)
        self.cqt = CQT1992v2(sr=sample_rate, hop_length=hop_length, fmin=fmin, fmax=None, n_bins=n_bins,
                            bins_per_octave=bins_per_octave, norm=1, window="hann", center=True, pad_mode="reflect")
                
    def forward(self, x):
        self.cqt = self.cqt.to(x.device)
        out = self.cqt(x)
        return out


class MidiERB(nn.Module):
    """
    Similar to CQT but using ERB triangle filters on spectrograms.
    
    Reference:
            Data-Driven Harmonic Filters for Audio Representation Learning (Won et al., 2020)
            https://github.com/minzwon/sota-music-tagging-models/blob/master/training/modules.py

    Args:
            lowest_hz (float): the lowest frequency (Default: 65.4064, i.e., 'C2')
            levels (int): number of output frequency bins (Default: 128)
            alpha (float): ERB parameter (Default: 0.1079)
            beta (float): ERB parameter (Default: 24.7)
            sample_rate (int): sampling rate (Default: 16000)
            n_bins (int): number of input frequency bins (Default: 257)
    Input:
            spec (torch.tensor): input spectrogram (batch, n_bins, length)
    Returns:
            out (torch.tensor): output tensor similar to CQT. Power_to_DB required.
    """
    def __init__(self,
                 lowest_hz=65.40639132514966,
                 levels=128,
                 alpha=0.1079,
                 beta=24.7,
                 sample_rate=16000,
                 n_bins=257):
        super(MidiERB, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.init_hz(lowest_hz, levels, sample_rate, n_bins)
        self.register_buffer('fb', self.init_filterbank())

    @staticmethod
    def hz_to_midi(freq):
        return 12 * (np.log2(np.asanyarray(freq)) - np.log2(440.0)) + 69

    @staticmethod
    def midi_to_hz(notes):
        return 440.0 * (2.0 ** ((np.asanyarray(notes) - 69.0) / 12.0))

    def init_hz(self, lowest_hz, levels, sample_rate, n_bins):
        # upper and lower bounds
        lowest_midi = self.hz_to_midi(lowest_hz)
        highest_midi = self.hz_to_midi(sample_rate / 2)

        # get midi scale
        midi = np.linspace(lowest_midi, highest_midi, levels + 1)
        hz = self.midi_to_hz(midi[:-1])
        self.f0 = torch.tensor(hz.astype('float32'))
        self.fft_bins = torch.linspace(0, sample_rate//2, n_bins)

    def init_filterbank(self):
        bw = self.alpha * self.f0 + self.beta
        bw = bw.unsqueeze(0)
        f0 = self.f0.unsqueeze(0)
        fft_bins = self.fft_bins.unsqueeze(1)

        up_slope = torch.matmul(fft_bins, (2/bw)) + 1 - (2*f0/bw)
        down_slope = torch.matmul(fft_bins, (-2/bw)) + 1 + (2*f0/bw)
        fb = torch.max(torch.zeros(1), torch.min(down_slope, up_slope))
        return fb

    def forward(self, spec):
        out = torch.matmul(spec.transpose(1, 2), self.fb).transpose(1, 2)
        return out


class PseudoPitchShift(nn.Module):
    """
    Motivated from Google SPICE.
    Crop certain frequency bins to expect the effects of pitch shift.

    Args:
        n_bins (int): number of output bins (Default: 128)
        margin (int): maximum bins of pitch shift (Default: 10)
    Input:
        cqt (torch.tensor): input CQT [batch, n_bins + (2*margin), time]
    Returns:
        out (torch.tensor): pitch-shifted output tensor [batch, n_bins, time]
    """
    def __init__(self,
                 n_bins=128,
                 margin=10,
                 is_shift=False):
        super(PseudoPitchShift, self).__init__()
        self.n_bins = n_bins
        self.margin = margin
        self.is_shift = is_shift

    def shuffle_cqt(self, x):
        b, f, t = x.size()
        new_x = torch.zeros(b, self.n_bins, t)
        for _b in range(b):
            rand_freq = int(random.uniform(0, (self.margin * 2 + 1)))
            new_x[_b] = x[_b, rand_freq:rand_freq+self.n_bins]
        new_x = new_x.to(x.device)
        return new_x

    def crop_cqt(self, x):
        return x[:, self.margin:-self.margin]

    def forward(self, x):
        if self.is_shift:
            return self.shuffle_cqt(x)
        else:
            return self.crop_cqt(x)


#######################################################
# Data augmentation from the field of computer vision #
#######################################################
class GaussianNoise2D(nn.Module):
    """
    Adding Gaussian noise to the 2-dimensional representation. This module treats audio like a 2D image.

    Args:
            ratio (float)
    Input:
            spec (torch.tensor): input spectrogram(batch, n_bins, length)
    Returns:
            out (torch.tensor): noise added spectrogram
    """
    def __init__(self,
                 ratio=0.0):
        super(GaussianNoise2D, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        noise = torch.zeros(x.size())
        noise = noise.normal_(0, 1) * self.ratio * x.max()
        noise = noise.to(x.device)
        return x + noise


class GaussianBlur(nn.Module):
    """
    Gaussian Blur
    [TODO]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    """
    def __init__(self):
        super(GaussianBlur, self).__init__()

    def forward(self, x):
        return 0


class TimeMask(nn.Module):
    """
    Mask time bins.

    Args:
        batch (int): batch size
        ratio (float): masking ratio
    Input:
        spec (torch.tensor): input spectrogram(batch, n_bins, length)
    Returns:
        out (torch.tensor): temporally masked spectrogram
    """
    def __init__(self,
                 batch=64,
                 ratio=0.0,
                 mask_value=-80):
        super(TimeMask, self).__init__()
        self.batch= batch
        self.ratio = ratio
        self.mask_value = mask_value

    def get_rand(self, x):
        width = torch.zeros(self.batch)
        ix = torch.zeros(self.batch)
        max_width = int(x.size(-1) * self.ratio)

        width = width.uniform_(0, max_width)
        ix = ix.uniform_(0, x.size(-1)-max_width)
        return width, ix

    def get_mask(self, x, width, ix):
        mask = torch.ones(x.size())
        for _b in range(mask.size(0)):
            x[_b, :, int(ix[_b]):int(ix[_b]+width[_b])] = self.mask_value
        return x

    def forward(self, x):
        width, ix = self.get_rand(x)
        out = self.get_mask(x, width, ix)
        return out


class FreqMask(nn.Module):
    """
    Mask frequency bins.

    Args:
        batch (int): batch size
        n_bins (int): number of requency bins
        ratio (float): masking ratio
    Input:
        spec (torch.tensor): input spectrogram(batch, n_bins, length)
    Returns:
        out (torch.tensor): temporally masked spectrogram
    """
    def __init__(self,
                 batch=64,
                 n_bins=128,
                 ratio=0.0,
                 mask_value=-80):
        super(FreqMask, self).__init__()
        self.batch = batch
        self.n_bins = n_bins
        self.ratio = ratio
        self.mask_value = mask_value
        self.max_width = int(n_bins * ratio)

    def get_rand(self, x):
        width = torch.zeros(self.batch)
        ix = torch.zeros(self.batch)

        width = width.uniform_(0, self.max_width)
        ix = ix.uniform_(0, self.n_bins-self.max_width)
        return width, ix

    def get_mask(self, x, width, ix):
        for _b in range(x.size(0)):
            x[_b, int(ix[_b]):int(ix[_b]+width[_b])] = self.mask_value
        return x

    def forward(self, x):
        width, ix = self.get_rand(x)
        out = self.get_mask(x, width, ix)
        return out


class CutOut(nn.Module):
    """
    Cut out with a squared mask.
    [TODO]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def __init__(self):
        super(CutOut, self).__init__()

    def forward(self, x):
        return 0


######################################
# Data augmentation by mixing labels #
######################################
class MixUp(nn.Module):
    """
    [TODO]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def __init__(self):
        super(MixUp, self).__init__()

    def forward(self, x, y):
        return 0


class CutMix(nn.Module):
    """
    [TODO]!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    """
    def __init__(self):
        super(CutMix, self).__init__()

    def forward(self, x, y):
        return 0


#################################################
# get augmentation sequences for model training #
#################################################
def get_augmentation_sequence(config):
    augs = []
    no_augs = []

    # Gain (raw -> raw)
    if config.is_gain:
        augs.append(Gain(config.gain_db_min, config.gain_db_max))

    # Gaussian noise (raw -> raw)
    if config.is_noise:
        augs.append(GaussianNoise(config.noise_snr_min, config.noise_snr_max))

    # When output format is mel spectrogram.
    if config.output_type == "spec":
        # STFT (raw -> complex spec)
        augs.append(STFT(n_fft=config.n_fft,
                         hop_length=config.hop_length,
                         win_length=config.win_length))
        no_augs.append(STFT(n_fft=config.n_fft,
                            hop_length=config.hop_length,
                            win_length=config.win_length))

        # Complex norm (complex spec -> spec)
        augs.append(ComplexNorm(power=2.0))
        no_augs.append(ComplexNorm(power=2.0))

        # Mel scale (spec -> melspec)
        augs.append(MelScale(n_mels=config.n_bins,
                             sample_rate=config.sample_rate,
                             n_stft=config.n_fft//2+1))
        no_augs.append(MelScale(n_mels=config.n_bins,
                                sample_rate=config.sample_rate,
                                n_stft=config.n_fft//2+1))

        # Amplitude to db (spec -> spec)
        augs.append(AmplitudeToDB())
        no_augs.append(AmplitudeToDB())

    # When output format is CQT.
    elif config.output_type == "cqt":
        # CQT + Pseudo pitch shift (raw -> cqt)
        if config.is_pitch_shift:
            augs.append(CQT(sample_rate=config.sample_rate,
                            hop_length=config.hop_length,
                            n_bins=config.n_bins+(2*config.pitch_shift_margin)))
            no_augs.append(CQT(sample_rate=config.sample_rate,
                               hop_length=config.hop_length,
                               n_bins=config.n_bins+(2*config.pitch_shift_margin)))
            augs.append(PseudoPitchShift(n_bins=config.n_bins,
                                         margin=config.pitch_shift_margin,
                                         is_shift=True))
            no_augs.append(PseudoPitchShift(n_bins=config.n_bins,
                                            margin=config.pitch_shift_margin,
                                            is_shift=False))
        else:
            augs.append(CQT(sample_rate=config.sample_rate,
                            hop_length=config.hop_length,
                            n_bins=config.n_bins))
            no_augs.append(CQT(sample_rate=config.sample_rate,
                               hop_length=config.hop_length,
                               n_bins=config.n_bins))
        # Amplitude to db (cqt -> cqt)
        augs.append(AmplitudeToDB(stype='magnitude', top_db=80))
        no_augs.append(AmplitudeToDB(stype='magnitude', top_db=80))

    ## Exception error
    else:
        print("Invalid output type. Output type has to be 'raw', 'spec', or 'cqt'.")


    # Data augmentation for 2D representations
    # Gaussian noise 2D
    if config.is_noise2d:
        augs.append(GaussianNoise2D(ratio=config.noise2d_ratio))

    # Gaussian blur

    # Time masking
    if config.is_time_mask:
        augs.append(TimeMask(batch=config.batch_size,
                             ratio=config.time_mask_ratio))

    # Frequency masking
    if config.is_freq_mask:
        augs.append(FreqMask(batch=config.batch_size,
                             n_bins=config.n_bins,
                             ratio=config.freq_mask_ratio))

    # sequential
    aug_seq = nn.Sequential(*augs)
    eval_seq = nn.Sequential(*no_augs)

    return aug_seq, eval_seq


###############################################
# get an augmentation sequence for evaluation #
###############################################
def get_eval_sequence(config):
    augs = []

    # Gain (raw -> raw)
    if config.is_gain:
        augs.append(Gain(config.gain_db_min, gain_db_max))

    # Gaussian noise (raw -> raw)
    if config.is_noise:
        augs.append(GaussianNoise(config.noise_snr_min, config.noise_snr_max))

    # When output format is raw waveform.
    if config.output_type == "raw":
        # STFT (raw -> complex spec)
        augs.append(STFT(n_fft=config.n_fft,
                        hop_length=config.hop_length,
                        win_length=config.win_length))

        # Time stretch (complex spec -> complex spec)
        if config.is_time_stretch:
            augs.append(TimeStretch(hop_length=config.hop_length,
                                    n_freq=config.n_fft//2+1,
                                    rate_min=config.time_stretch_min,
                                    rate_max=config.time_stretch_max))

        # Pitch shift (complex spec -> complex spec)
        if config.is_pitch_shift:
            augs.append(PitchShift(input_length=config.input_length,
                                   sample_rate=config.sample_rate,
                                   n_fft=config.n_fft,
                                   win_length=config.win_length,
                                   hop_length=config.hop_length,
                                   n_freq=config.n_fft//2+1,
                                   margin=config.pitch_shift_margin,
                                   is_random=False))

        # ISTFT (complex spec -> raw)
        augs.append(ISTFT(n_fft=config.n_fft,
                          hop_length=config.hop_length,
                          win_length=config.win_length))

    # When output format is mel spectrogram.
    elif config.output_type == "spec":
        # STFT (raw -> complex spec)
        augs.append(STFT(n_fft=config.n_fft,
                         hop_length=config.hop_length,
                         win_length=config.win_length))

        # Time stretch (complex spec -> complex spec)
        if config.is_time_stretch:
            augs.append(TimeStretch(hop_length=config.hop_length,
                                    n_freq=config.n_fft//2+1,
                                    rate_min=config.time_stretch_min,
                                    rate_max=config.time_stretch_max))

        # Pitch shift (complex spec -> complex spec)
        if config.is_pitch_shift:
            augs.append(PitchShift(input_length=config.input_length,
                                   sample_rate=config.sample_rate,
                                   n_fft=config.n_fft,
                                   win_length=config.win_length,
                                   hop_length=config.hop_length,
                                   n_freq=config.n_fft//2+1,
                                   margin=config.pitch_shift_margin,
                                   is_random=False))

        # Complex norm (complex spec -> spec)
        augs.append(ComplexNorm(power=2.0))

        # Mel scale (spec -> melspec)
        augs.append(MelScale(n_mels=config.n_bins,
                             sample_rate=config.sample_rate,
                             n_stft=config.n_fft//2+1))

    # When output format is CQT.
    elif config.output_type == "cqt":
        # CQT + Pseudo pitch shift (raw -> cqt)
        if config.is_pitch_shift:
            augs.append(CQT(sample_rate=config.sample_rate,
                            hop_length=config.hop_length,
                            n_bins=config.n_bins+(2*config.pitch_shift_margin)))
            augs.append(PseudoPitchShift(n_bins=config.n_bins,
                                         margin=config.pitch_shift_margin,
                                         is_shift=False))
        else:
            augs.append(CQT(sample_rate=config.sample_rate,
                            hop_length=config.hop_length,
                            n_bins=config.n_bins))
        # Amplitude to db (cqt -> cqt)
        augs.append(AmplitudeToDB(stype='magnitude', top_db=80))
        no_augs.append(AmplitudeToDB(stype='magnitude', top_db=80))

    ## Exception error
    else:
        print("Invalid output type. Output type has to be 'raw', 'spec', or 'cqt'.")


    # Data augmentation for 2D representations
    # Gaussian noise 2D
    if config.is_noise2d:
        augs.append(GaussianNoise2D(ratio=config.noise2d_ratio))

    # Time masking
    if config.is_time_mask:
        augs.append(TimeMask(batch=config.batch_size,
                             ratio=config.time_mask_ratio))

    # Frequency masking
    if config.is_freq_mask:
        augs.append(FreqMask(batch=config.batch_size,
                             n_bins=config.n_bins,
                             ratio=config.freq_mask_ratio))

    # sequential
    eval_seq = nn.Sequential(*augs)

    return eval_seq

