import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import scaper
import soundfile as sf
import os
from os.path import join
import librosa
import random
import argparse
import json
from math import floor
import sed_eval
from sed_eval import sound_event
import sed_eval.metric as sed


parser = argparse.ArgumentParser(description='Scaper Sound Simulator')
parser.add_argument('--count', type=str, help='Sound event count', required=True)
parser.add_argument('--snr', type=str, help='Reference dB', default=-33)
parser.add_argument('--fore', type=str, help='Foreground sound label', default='bird')
parser.add_argument('--back', type=str, help='Annotation file name', default='park')
parser.add_argument('--samples', type=str, help='Number of samples per settings', default=10)

args = parser.parse_args()

root_dir = '/home/zdai/repos/pyroomacoustics/pra_workspace'
#root_dir = '/Users/zhuangzhuangdai/repos/pyroomacoustics/pra_workspace'

class BirdInstance(object):
    def __init__(self, seed, xyz, origin, t, snr=1.):
        self.seed = seed
        self.origin = origin

        if snr != 1.:
            pass
        self.signal = self.seed['signal']

        x = random.uniform(0, xyz[0])
        y = random.uniform(0, xyz[1])
        z = random.uniform(0, xyz[2])
        self.pos_3D = np.array([x, y, z])

        if self.seed['name'] == 'eagle' or self.seed['name'] == 'rooster':
            while np.linalg.norm(self.pos_3D - self.origin) < 40:
                x = random.uniform(0, xyz[0])
                y = random.uniform(0, xyz[1])
                z = random.uniform(0, xyz[2])
                self.pos_3D = np.array([x, y, z])
        else:
            while np.linalg.norm(self.pos_3D - self.origin) < 5 or np.linalg.norm(self.pos_3D - self.origin) > 60:
                x = random.uniform(0, xyz[0])
                y = random.uniform(0, xyz[1])
                z = random.uniform(0, xyz[2])
                self.pos_3D = np.array([x, y, z])

        # No need for delay penalty at end of clip ?!
        self.delay = (t - self.seed['len']) * random.random()
        #self.delay = float(t) * random.random()

        # Round delay to discrete 100ms
        self.delay = round(self.delay, 1)

    def to_dict(self):
        BirdDict = {'BirdName': self.seed['name'],
                    'location': self.pos_3D.tolist(),
                    'start_t': self.delay,
                    'end_t': self.delay + self.seed['len']}
        return BirdDict


class ScpCrowd(object):
    def __init__(self, seeds: list, count: int, room_size: np.array, snr=(-33., 2.), output_filename=None,
                 fs=22050, max_order=3, absorb=0.2, noise_pos=None, micro_pos=None,
                 temporal_density=None, spectro_density=None, idx=0):
        # Random Clip Length
        #self.clip_t = random.randint(3, 7)
        # Choose clip length in list
        length_choices = [5]
        self.clip_t = random.choice(length_choices)

        print("This clip lasts %.3f s" % self.clip_t)
        self.count = count
        self.density = float(self.count / self.clip_t)
        self.PR = -1.
        self.PD = -1.

        self.room_size = room_size
        self.wall_absorption = absorb
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
                                          materials=pra.Material(self.wall_absorption, 0.15),
                                          ray_tracing=True,
                                          air_absorption=True,
                                          sigma2_awgn=self.sigma2_awgn)
        self.room.extrude(self.height, materials=pra.Material(self.wall_absorption, 0.15))

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
                               xyz=(self.room_size[0], self.room_size[1], self.height),
                               origin=self.micro_pos, t=self.clip_t)
            self.sound_srcs.append(src.to_dict())
            self.room.add_source(src.pos_3D, signal=src.signal, delay=src.delay)

    def get_seed_sound_info(self, seed, sr):
        soundname = seed.split('.')[0].split('/')[-1]
        fs, signal = wavfile.read(seed)

        # Up/Down Sampling
        if fs != sr:
            signal, fs = librosa.load(seed, sr=sr)
            signal = (signal * 32767).astype(int)

        # Clip the seed length to 100ms
        floor_length = floor(len(signal) / fs * 10) / 10
        signal = signal[:int(floor_length * fs)]

        self.seed_sound_info[soundname] = dict()
        self.seed_sound_info[soundname]['wavfile'] = seed
        self.seed_sound_info[soundname]['signal'] = signal
        self.seed_sound_info[soundname]['fs'] = fs
        self.seed_sound_info[soundname]['len'] = floor_length
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
                             'wall_absorption': self.wall_absorption,
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
        # TODO: inherit from beamforming.py rather than manipulate source code
        self.room.mic_array.to_wav_t(join(root_dir, 'outputs', self.wavfile_savename + ".wav"),
                                     t=self.clip_t,
                                     norm=False,
                                     bitdepth=np.int16)
        '''
        self.room.mic_array.to_wav(join(root_dir, 'outputs', self.wavfile_savename + ".wav"),
                                     #t=self.clip_t,
                                     norm=False,
                                     bitdepth=np.int16)
        '''

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
    path_to_audio = join(root_dir, 'audio')

    outdir = 'scaper_outs'
    os.makedirs(join(root_dir, outdir)) if not os.path.exists(join(root_dir, outdir)) else None
    # Annotation file
    annfile = join(root_dir, outdir, 'scp_anns.json')

    event_count = int(args.count)
    clip_length = 5.0
    seed = 123
    foreground_folder = os.path.join(path_to_audio, 'foreground')
    background_folder = os.path.join(path_to_audio, 'background')

    #sampling_rate = 44100
    sampling_rate = 24000

    num_samples = int(args.samples)
    anns = {}

    for index in range(num_samples):
        filename = "scp_count-{}_index-{}".format(args.count, index)

        audiofile = join(outdir, filename + '.wav')
        jamsfile = join(outdir, filename + '.jams')
        txtfile = join(outdir, filename + '.txt')

        sc = scaper.Scaper(duration=clip_length,
                           fg_path=foreground_folder,
                           bg_path=background_folder,
                           #random_state=seed,
                           )
        sc.sr = sampling_rate
        sc.n_channels = 1
        sc.ref_db = int(args.snr)

        sc.add_background(label=('const', args.back),
                          source_file=('choose', []),
                          source_time=('uniform', 0, 30))

        for _ in range(event_count):
            sc.add_event(label=('const', args.fore),
                         source_file=('choose', []),
                         source_time=('const', 0),
                         event_time=('uniform', 0, sc.duration * 0.8),
                         event_duration=('const', 1000),  # Event_duration > seed_length guarantee Full seed
                         snr=('normal', 5, 5),
                         pitch_shift=('uniform', -2, 2),
                         time_stretch=('uniform', 0.8, 1.2))

        # Noise Events
        for _ in range(3):
            sc.add_event(label=('choose', ['siren', 'car_horn']),
                         source_file=('choose', []),
                         source_time=('const', 0),
                         event_time=('uniform', 0, sc.duration * 0.8),
                         event_duration=('truncnorm', 1, 0.3, sc.duration * 0.1, sc.duration * 0.9),
                         snr=('normal', 3, 3),
                         pitch_shift=('uniform', -2, 2),
                         time_stretch=('uniform', 0.8, 1.2))

        mixture_audio, mixture_jam, annotation_list, stem_audio_list = sc.generate(audiofile,
                                                                                   #jamsfile,
                                                                                   allow_repeated_label=True,
                                                                                   allow_repeated_source=True,
                                                                                   reverb=0.1,
                                                                                   disable_sox_warnings=True,
                                                                                   no_audio=False,
                                                                                   #txt_path=txtfile,
                                                                                   fix_clipping=True,
                                                                                   quick_pitch_time=False,)
        #print(annotation_list, stem_audio_list)

        anns[filename] = annotation_list

    if os.path.exists(annfile):
        with open(annfile) as f:
            new_anns = json.load(f)

        new_anns.update(anns)
        with open(annfile, 'w', encoding='utf-8') as f:
            json.dump(new_anns, f, ensure_ascii=False, indent=4)
    else:
        with open(annfile, 'w', encoding='utf-8') as f:
            json.dump(anns, f, ensure_ascii=False, indent=4)

    print("Simulation Completed!")


if __name__ == '__main__':
    main()

