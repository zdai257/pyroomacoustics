import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import os
from os.path import join
import librosa
import random
import pandas as pd

root_dir = '/home/zdai/repos/pyroomacoustics/pra_workspace'


class BirdInstance(object):
    def __init__(self, seed, xyz, t, snr=1.):
        self.seed = seed

        if snr != 1.:
            pass
        self.signal = self.seed['signal']

        x = random.uniform(0, xyz[0])
        y = random.uniform(0, xyz[1])
        z = random.uniform(0, xyz[2])
        self.pos_3D = np.array([x, y, z])

        self.delay = (t - self.seed['len']) * random.random()

    def to_json(self):
        pass


class SoundCrowd(object):
    def __init__(self, seeds: list, count: int, snr: float, output_filename=None,
                 fs=24000, max_order=3, noise_pos=np.array([1., 1., 1.]),
                 temporal_density=None, spectro_density=None):
        self.seed_sound_info = dict()
        self.class_counter = 0
        self.fs = fs
        self.max_order = max_order
        self.noise_pos = noise_pos
        for item in seeds:
            self.get_seed_sound_info(item)
        print(self.seed_sound_info)

        self.clip_t = random.randint(5, 10)
        print("This clip lasts %.3f s" % self.clip_t)

        self.count = count
        # TODO: implement random SNR
        self.snr = snr

        if output_filename is not None:
            self.wavfile_savename = output_filename
        else:
            self.wavfile_savename = "Simulated_count{}.wav".format(self.count)

        # step1: build up the room
        self.corners = np.array([[0., 0.], [0., 5.], [7., 5.], [7., 0.]]).T  # [x,y]
        self.height = 3.
        self.room = pra.Room.from_corners(self.corners,
                                          fs=self.fs,
                                          max_order=self.max_order,
                                          materials=pra.Material(0.2, 0.15),
                                          ray_tracing=True,
                                          air_absorption=True)
        self.room.extrude(self.height, materials=pra.Material(0.2, 0.15))

        # Set the ray tracing parameters
        self.room.set_ray_tracing(receiver_radius=0.5,
                                  n_rays=10000,
                                  energy_thres=1e-5)

        # step2: add microphone array to the room
        R = np.array([[0.25], [0.25], [0.3]])
        self.room.add_microphone(loc=R, fs=self.fs)

        # step3: add sound source to the room
        # step3.1 Place a source of white noise playing for T s
        noise_amp = 1000
        self.noise_source = noise_amp * np.random.randn(self.fs * self.clip_t)
        self.room.add_source(self.noise_pos, signal=self.noise_source)

        # step3.2 Place Random Birds
        self.sound_srcs = []
        max_xy = np.max(self.corners, axis=0)
        for index in range(count):
            seedname = random.choice(list(self.seed_sound_info.keys()))
            src = BirdInstance(self.seed_sound_info[seedname], xyz=(max_xy[0], max_xy[1], self.height), t=self.clip_t)
            self.sound_srcs.append(src)
            self.room.add_source(src.pos_3D, signal=src.signal, delay=src.delay)


    def get_seed_sound_info(self, seed):
        soundname = seed.split('.')[0].split('/')[-1]
        fs, signal = wavfile.read(seed)

        self.seed_sound_info[soundname] = dict()
        self.seed_sound_info[soundname]['wavfile'] = seed
        self.seed_sound_info[soundname]['signal'] = signal
        self.seed_sound_info[soundname]['fs'] = fs
        self.seed_sound_info[soundname]['len'] = len(signal) / fs
        self.seed_sound_info[soundname]['class_id'] = self.class_counter

        self.class_counter += 1

    def simulate(self):
        # compute image sources. At this point, RIRs are simply created
        self.room.image_source_model()
        self.room.simulate()

    def generate(self):
        self.room.mic_array.to_wav(join(root_dir, 'outputs', self.wavfile_savename),
                                   norm=False,
                                   bitdepth=np.int16)


def main():
    wav_lst = []
    wav_lst.append(join(root_dir, 'junco.wav'))
    wav_lst.append(join(root_dir, 'amre.wav'))

    highfidelity_samplerate = 44100

    sc = SoundCrowd(seeds=wav_lst, count=3, snr=1., fs=highfidelity_samplerate, max_order=3)

    sc.simulate()
    sc.generate()
    print("Simulation Completed!")


if __name__ == '__main__':
    main()

