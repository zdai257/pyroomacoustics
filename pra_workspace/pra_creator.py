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
import argparse
import json
import sed_eval
from sed_eval import sound_event
import sed_eval.metric as sed


parser = argparse.ArgumentParser(description='Room Sound Simulator')
parser.add_argument('--count', type=str, help='Sound event count', required=True)
parser.add_argument('--snr', type=str, help='Signal to Noise Ratio', default=-33)
parser.add_argument('--X', type=str, help='Side X length', default=50)
parser.add_argument('--Y', type=str, help='Side Y length', default=50)
parser.add_argument('--Z', type=str, help='Side Z length', default=10)
parser.add_argument('--rt_order', type=str, help='Max ray tracing order', default=3)
parser.add_argument('--samples', type=str, help='Number of samples per settings', default=10)
parser.add_argument('--annfile', type=str, help='Annotation file name', default='annotations.json')
args = parser.parse_args()

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

        # No need for delay penalty at end of clip ?!
        #self.delay = (t - self.seed['len']) * random.random()
        self.delay = float(t) * random.random()

    def to_dict(self):
        BirdDict = {'BirdName': self.seed['name'],
                    'location': self.pos_3D.tolist(),
                    'start_t': self.delay,
                    'end_t': self.delay + self.seed['len']}
        return BirdDict


class SoundCrowd(object):
    def __init__(self, seeds: list, count: int, room_size: np.array, snr=(-33., 2.), output_filename=None,
                 fs=22050, max_order=3, noise_pos=None, micro_pos=None,
                 temporal_density=None, spectro_density=None, idx=0):
        # Random Clip Length
        self.clip_t = random.randint(3, 7)
        print("This clip lasts %.3f s" % self.clip_t)
        self.count = count
        self.density = float(self.count / self.clip_t)
        self.PR = -1.
        self.PD = -1.

        self.room_size = room_size
        # Randomize SNR
        self.snr = random.normalvariate(mu=snr[0], sigma=snr[1])
        # Compute the variance of the microphone noise
        self.sigma2_awgn = 10 ** (-self.snr / 10) * 1

        self.seed_sound_info = dict()
        self.class_counter = 0
        self.fs = fs
        self.max_order = max_order
        self.noise_pos = noise_pos
        if micro_pos is not None:
            self.micro_pos = micro_pos
        else:
            self.micro_pos = np.array([[random.uniform(0, room_size[0])],
                                       [random.uniform(0, room_size[1])],
                                       [random.uniform(0, room_size[2])]])

        for item in seeds:
            self.get_seed_sound_info(item, self.fs)
        #print(self.seed_sound_info)

        if output_filename is not None:
            self.wavfile_savename = output_filename
        else:
            self.wavfile_savename = "No-{}_ClipLength{}_Count{}".format(idx, self.clip_t, self.count)

        # step1: build up the room (assume Rectangular)
        self.corners = np.array([[0., 0.], [0., room_size[1]], [room_size[0], room_size[1]], [room_size[0], 0.]]).T
        self.height = room_size[2]
        self.room = pra.Room.from_corners(self.corners,
                                          fs=self.fs,
                                          max_order=self.max_order,
                                          materials=pra.Material(0.2, 0.15),
                                          ray_tracing=True,
                                          air_absorption=True,
                                          sigma2_awgn=self.sigma2_awgn)
        self.room.extrude(self.height, materials=pra.Material(0.2, 0.15))

        # Set the ray tracing parameters
        self.room.set_ray_tracing(receiver_radius=0.5,
                                  n_rays=10000,
                                  energy_thres=1e-5)

        # step2: add microphone array to the room
        self.room.add_microphone(loc=self.micro_pos, fs=self.fs)

        # step3: add sound source to the room
        # step3.1 Place a source of white noise playing for T s
        '''
        noise_amp = 300
        self.noise_source = noise_amp * np.random.randn(self.fs * self.clip_t)
        if self.noise_pos is not None:
            self.room.add_source(self.noise_pos, signal=self.noise_source)
        else:
            self.room.add_source(room_size / 2., signal=self.noise_source)
        '''

        # step3.2 Place Random Birds
        self.sound_srcs = []

        for index in range(self.count):
            seedname = random.choice(list(self.seed_sound_info.keys()))
            src = BirdInstance(self.seed_sound_info[seedname],
                               xyz=(self.room_size[0], self.room_size[1], self.height), t=self.clip_t)
            self.sound_srcs.append(src.to_dict())
            self.room.add_source(src.pos_3D, signal=src.signal, delay=src.delay)

    def get_seed_sound_info(self, seed, sr):
        soundname = seed.split('.')[0].split('/')[-1]
        fs, signal = wavfile.read(seed)

        # Up/Down Sampling
        if fs != sr:
            signal, fs = librosa.load(seed, sr=sr)
            signal = (signal * 32767).astype(int)

        self.seed_sound_info[soundname] = dict()
        self.seed_sound_info[soundname]['wavfile'] = seed
        self.seed_sound_info[soundname]['signal'] = signal
        self.seed_sound_info[soundname]['fs'] = fs
        self.seed_sound_info[soundname]['len'] = len(signal) / fs
        self.seed_sound_info[soundname]['name'] = soundname
        self.seed_sound_info[soundname]['class_id'] = self.class_counter

        self.class_counter += 1

    def to_dict(self):
        item = {
                             'room_size': self.room_size.tolist(),
                             'fs': self.fs,
                             'snr': self.snr,
                             'sigma2_awgn': self.sigma2_awgn,
                             'max_order': self.max_order,
                             'microphone_pos': self.micro_pos.tolist(),
                             'clip_length': self.clip_t,
                             'count': self.count,
                             'birds': self.sound_srcs,
                             'PolyphonicRatio': self.PR,
                             'PolyphonicDensity': self.PD,
                }
        return item

    def simulate(self):
        # compute image sources. At this point, RIRs are simply created
        self.room.image_source_model()

        #self.room.simulate(snr=random.gauss(self.snr[0], self.snr[1]))
        self.room.simulate()

    def generate(self):
        os.makedirs(join(root_dir, 'outputs')) if not os.path.exists(join(root_dir, 'outputs')) else None
        self.room.mic_array.to_wav_t(join(root_dir, 'outputs', self.wavfile_savename + ".wav"),
                                     t=self.clip_t,
                                     norm=False,
                                     bitdepth=np.int16)

    def polyphony(self):
        step = 0.0001

        t = np.arange(0, self.clip_t, step)
        t_tp = np.zeros(t.shape)
        t_den = np.zeros(t.shape)

        for idx, x in np.ndenumerate(t):
            mono_track = 0
            for instance in self.sound_srcs:
                if x >= instance['start_t'] and x <= instance['end_t']:
                    mono_track += 1

            if mono_track >= 2:
                t_tp[idx] = step
                t_den[idx] = mono_track * step

        polyphonic_ratio = np.sum(t_tp) / self.clip_t
        polyphonic_density = np.sum(t_den) / (self.clip_t * self.count)

        self.PR = polyphonic_ratio
        self.PD = polyphonic_density


def main():
    wav_lst = []
    wav_lst.append(join(root_dir, 'junco.wav'))
    wav_lst.append(join(root_dir, 'amre.wav'))
    wav_lst.append(join(root_dir, 'duck.wav'))
    wav_lst.append(join(root_dir, 'eagle.wav'))
    wav_lst.append(join(root_dir, 'japanrobin.wav'))

    #sampling_rate = 44100
    sampling_rate = 22050

    room_size = np.array([float(args.X), float(args.Y), float(args.Z)], dtype=float)
    assert isinstance(int(args.X), int)

    num_samples = int(args.samples)
    anns = {}

    for index in range(num_samples):
        filename = "X-{}_Y-{}_count-{}_index-{}".format(int(args.X), int(args.Y), args.count, index)

        sc = SoundCrowd(seeds=wav_lst, room_size=room_size, count=int(args.count), snr=(float(args.snr), 2.),
                        fs=sampling_rate, max_order=int(args.rt_order), idx=index,
                        output_filename=filename)
        sc.simulate()
        sc.generate()
        sc.polyphony()

        anns[filename] = sc.to_dict()

    if os.path.exists(join(root_dir, args.annfile)):
        with open(args.annfile) as f:
            new_anns = json.load(f)

        new_anns.update(anns)
        with open(args.annfile, 'w', encoding='utf-8') as f:
            json.dump(new_anns, f, ensure_ascii=False, indent=4)
    else:
        with open(args.annfile, 'w', encoding='utf-8') as f:
            json.dump(anns, f, ensure_ascii=False, indent=4)
    print("Simulation Completed!")


if __name__ == '__main__':
    main()

