import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import fftconvolve
import IPython
import pyroomacoustics as pra
import os
import librosa
import random
import pandas as pd


def simulate_room_audio(soundobj_list, place_microphone_center=True, wavfile_savename=None):
    # step1: build up the room
    corners = np.array([[0., 0.], [0., 5.], [7., 5.], [7., 0.]]).T  # [x,y]
    room = pra.Room.from_corners(corners,
                                 fs=24000,
                                 max_order=3,
                                 materials=pra.Material(0.2, 0.15),
                                 ray_tracing=True,
                                 air_absorption=True)
    room.extrude(3., materials=pra.Material(0.2, 0.15))

    fig, ax = room.plot()
    ax.set_xlim([0, 7.])
    ax.set_ylim([0, 5.])
    ax.set_zlim([0, 3.])
    plt.show()

    # Set the ray tracing parameters
    room.set_ray_tracing(receiver_radius=0.5,
                         n_rays=10000,
                         energy_thres=1e-5)

    # step2: add microphone array to the room
    if place_microphone_center:
        R = np.array([[3.45, 3.55, 3.45, 3.55], [2.45, 2.45, 2.55, 2.55], [1.5, 1.5, 1.5, 1.5]])
    else:
        R = np.array([[0.25, 0.35, 0.25, 0.35], [0.25, 0.25, 0.35, 0.35], [0.3, 0.3, 0.3, 0.3]])

    room.add_microphone(loc=R, fs=24000)

    # step3: add sound source to the room
    for idx, soundobj_tmp in enumerate(soundobj_list):
        if idx > 0:
            break
        # pos_3D = [6.9,4.9,2.9]
        pos_3D = [1., 1., 1.]
        fs, signal = wavfile.read('./wavfile/baby_mama_24k_8s.wav')
        room.add_source(pos_3D,
                        signal=signal,
                        delay=0.)

    # compute image sources. At this point, RIRs are simply created
    room.image_source_model()

    room.simulate()

    wavfile_savename = '/Users/yuhhe/Downloads/test_near_babymama_1.wav'
    room.mic_array.to_wav(wavfile_savename,
                          norm=False,
                          bitdepth=np.int16)


def get_one_random_loc():
    # x_range = [0.1, 6.9]
    # y_range = [0.1, 4.9]
    # z_range = [0.1, 2.9]

    x_range = [2.5, 4.0]
    y_range = [1.5, 3.0]
    z_range = [1.45, 1.55]

    x_avoid = [0.3, 3.5]
    y_avoid = [0.3, 2.5]
    z_avoid = [0.3, 1.5]

    x = random.uniform(a=x_range[0], b=x_range[1])
    y = random.uniform(a=y_range[0], b=y_range[1])
    z = random.uniform(a=z_range[0], b=z_range[1])

    if abs(x - x_avoid[0]) <= 0.2:
        x = x_avoid[0] + 0.2
    elif abs(x - x_avoid[1]) <= 0.2:
        x = x_avoid[1] + 0.2

    if abs(y - y_avoid[0]) <= 0.2:
        y = y_avoid[0] + 0.2
    elif abs(y - y_avoid[1]) <= 0.2:
        y = y_avoid[1] + 0.2

    if abs(z - z_avoid[0]) <= 0.2:
        z = z_avoid[0] + 0.2
    elif abs(z - z_avoid[1]) <= 0.2:
        z = z_avoid[1] + 0.2

    return [x, y, z]


def get_seed_sound_info():
    seed_sound_info = dict()

    seed_sound_info['cat'] = dict()
    seed_sound_info['cat']['len'] = 4
    seed_sound_info['cat']['wavfile'] = './wavfile/07045201_cat_24k_4s.wav'
    seed_sound_info['cat']['class_id'] = 0

    seed_sound_info['boy_sing'] = dict()
    seed_sound_info['boy_sing']['len'] = 10
    seed_sound_info['boy_sing']['wavfile'] = './wavfile/boy_singing_24k_10s.wav'
    seed_sound_info['boy_sing']['class_id'] = 1

    seed_sound_info['baby_mama'] = dict()
    seed_sound_info['baby_mama']['len'] = 8
    seed_sound_info['baby_mama']['wavfile'] = './wavfile/baby_mama_24k_8s.wav'
    seed_sound_info['baby_mama']['class_id'] = 2

    seed_sound_info['dog_barking'] = dict()
    seed_sound_info['dog_barking']['len'] = 10
    seed_sound_info['dog_barking']['wavfile'] = './wavfile/dog_barking_24k_10s.wav'
    seed_sound_info['dog_barking']['class_id'] = 3

    seed_sound_info['piano_tap_dance'] = dict()
    seed_sound_info['piano_tap_dance']['len'] = 10
    seed_sound_info['piano_tap_dance']['wavfile'] = './wavfile/tap-dancer-piano_24k_10s.wav'
    seed_sound_info['piano_tap_dance']['class_id'] = 4

    return seed_sound_info


def update_one_track(start_frame, end_frame, class_id, pos_3D, track_id, input_track):
    [x, y, z] = pos_3D
    for frame_idx in range(start_frame, end_frame + 1):
        input_track[frame_idx, :] = [frame_idx, class_id, track_id, x, y, z]

    return input_track


def combine_two_tracks(first_track, second_track):
    '''
    combine two trackes
    :param first_track:
    :param second_track:
    :return: combined tracks
    '''
    combined_track = np.zeros(shape=[0, 6], dtype=np.float32)

    for row_idx in range(first_track.shape[0]):
        if first_track[row_idx, 0] != -1.:
            combined_track = np.vstack([combined_track, first_track[row_idx, :]])
        if second_track[row_idx, 0] != -1.:
            combined_track = np.vstack([combined_track, second_track[row_idx, :]])

    return combined_track


def get_one_episode_config():
    '''
    the first track only contains non-overlap soundobj
    the second track only contains overlapping soundobj
    :return:
    '''
    seed_sound_info = get_seed_sound_info()
    key_list = ['cat', 'boy_sing', 'baby_mama', 'dog_barking', 'piano_tap_dance']

    # initialize output
    constructed_soundobj = list()

    # initialize the two track csv file
    first_track = -1 * np.ones(shape=[600, 6], dtype=np.float32)  # [frame_idx, class_id, track_id, x, y, z]
    second_track = -1 * np.ones(shape=[600, 6], dtype=np.float32)
    track1_start_frame = 0
    track1_end_frame = 0
    track2_start_frame = 0
    track2_end_frame = 0

    # step1: first get a random sound as the start sound
    key_name = key_list[random.randint(a=0, b=len(key_list) - 1)]
    pos_3D = get_one_random_loc()

    soundobj_tmp = dict()
    soundobj_tmp['wavfile'] = seed_sound_info[key_name]['wavfile']
    soundobj_tmp['pos_3D'] = pos_3D
    soundobj_tmp['delay_time'] = 0
    soundobj_tmp['start_frame'] = track1_start_frame
    soundobj_tmp['end_frame'] = track1_start_frame + 10 * seed_sound_info[key_name]['len']
    soundobj_tmp['track_id'] = 0
    constructed_soundobj.append(soundobj_tmp)
    first_track = update_one_track(start_frame=soundobj_tmp['start_frame'],
                                   end_frame=soundobj_tmp['end_frame'],
                                   class_id=seed_sound_info[key_name]['class_id'],
                                   track_id=0,
                                   pos_3D=pos_3D,
                                   input_track=first_track)

    track1_end_frame = soundobj_tmp['end_frame']
    track1_start_frame = soundobj_tmp['start_frame']

    add_num = 0

    while True:
        # randomly choose a soundobj
        # key_name = key_list[random.randint(a=0, b=len(key_list) - 1)]
        if add_num >= 1:
            break
        add_num += 1
        key_name = key_list[0]
        pos_3D = get_one_random_loc()

        soundobj_tmp = dict()
        soundobj_tmp['wavfile'] = seed_sound_info[key_name]['wavfile']
        soundobj_tmp['pos_3D'] = pos_3D

        # decide where to put this soubdobj
        # add_overlap = random.randint(a=0, b=2) == 0
        add_overlap = False

        if add_overlap:
            add_start_frame = max((track1_start_frame + track1_end_frame) // 2, track2_end_frame + 1)
            add_end_frame = min(add_start_frame + seed_sound_info[key_name]['len'] * 10, 599)
            soundobj_tmp['start_frame'] = add_start_frame
            soundobj_tmp['end_frame'] = add_end_frame
            soundobj_tmp['track_id'] = 1
            soundobj_tmp['delay_time'] = float(add_start_frame - 1) / 10.

            track2_start_frame = add_start_frame
            track2_end_frame = add_end_frame

            track1_start_frame = add_end_frame + 1
            track1_end_frame = add_end_frame + 1

            second_track = update_one_track(start_frame=soundobj_tmp['start_frame'],
                                            end_frame=soundobj_tmp['end_frame'],
                                            class_id=seed_sound_info[key_name]['class_id'],
                                            track_id=1,
                                            pos_3D=pos_3D,
                                            input_track=second_track)

            constructed_soundobj.append(soundobj_tmp)

        else:
            add_start_frame = track1_end_frame + 1
            add_end_frame = min(add_start_frame + seed_sound_info[key_name]['len'] * 10, 599)
            soundobj_tmp['start_frame'] = add_start_frame
            soundobj_tmp['end_frame'] = add_end_frame
            soundobj_tmp['track_id'] = 1
            soundobj_tmp['delay_time'] = float(add_start_frame - 1) / 10.

            track1_start_frame = add_start_frame
            track1_end_frame = add_end_frame

            first_track = update_one_track(start_frame=soundobj_tmp['start_frame'],
                                           end_frame=soundobj_tmp['end_frame'],
                                           class_id=seed_sound_info[key_name]['class_id'],
                                           track_id=0,
                                           pos_3D=pos_3D,
                                           input_track=first_track)

            constructed_soundobj.append(soundobj_tmp)

        if track1_end_frame >= 599:
            break

    combined_track = combine_two_tracks(first_track, second_track)

    return constructed_soundobj, combined_track


def generate_episodes(episode_num=600,
                      csv_output_file_dir=None,
                      wav_center_output_dir=None,
                      wav_corner_output_dir=None):
    for episode_idx in range(episode_num):
        print('generating {}-th sound.'.format(episode_idx))
        constructed_soundobj, combined_track = get_one_episode_config()
        output_csv_filename = os.path.join(csv_output_file_dir, 'mix{}.csv'.format(episode_idx))
        output_wav_center_filename = os.path.join(wav_center_output_dir, 'mix{}.wav'.format(episode_idx))
        output_wav_corner_filename = os.path.join(wav_corner_output_dir, 'mix{}.wav'.format(episode_idx))

        pd.DataFrame(combined_track).to_csv(output_csv_filename,
                                            header=False,
                                            index=False,
                                            float_format='%.4f')

        # simulate_room_audio(constructed_soundobj,
        #                     place_microphone_center=True,
        #                     wavfile_savename=output_wav_center_filename)

        simulate_room_audio(constructed_soundobj,
                            place_microphone_center=False,
                            wavfile_savename=output_wav_corner_filename)

    print('Done!')

def main():
    episode_num = 1
    csv_output_dir = '/Users/yuhhe/Downloads/simulated_room_audio/meta_file/'
    wav_center_dir = '/Users/yuhhe/Downloads/simulated_room_audio/wav_center_file/'
    wav_corner_dir = '/Users/yuhhe/Downloads/simulated_room_audio/wav_corner_file/'
    os.makedirs(wav_center_dir) if not os.path.exists(wav_center_dir) else None
    os.makedirs(wav_corner_dir) if not os.path.exists(wav_corner_dir) else None
    os.makedirs(csv_output_dir) if not os.path.exists(csv_output_dir) else None

    generate_episodes(episode_num=episode_num,
                      csv_output_file_dir=csv_output_dir,
                      wav_center_output_dir=wav_center_dir,
                      wav_corner_output_dir=wav_corner_dir)

    print('Done!')


if __name__ == '__main__':
    main()