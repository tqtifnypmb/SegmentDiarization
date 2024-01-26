import os
import json
import wave
from dataclasses import dataclass

@dataclass
class Segment:
    speaker: str
    start: float
    end: float

def clean_labels():
    dir = "./vox_test/test_label"
    out_dir = "./vox_test/labels"
    talks = [file.split(".")[0] for file in os.listdir(dir)]
    files = list(set(talks))
    for talk in files:
        speakers = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        speakers_map = {}

        segments = []
        with open(os.path.join(dir, talk + ".rttm"), "r") as fp:
            for line in fp.readlines():
                comps = line.split(" ")

                begin_time = float(comps[3])
                len = float(comps[4])
                end_time = begin_time + len
                if comps[7] in speakers_map:
                    speaker_id = speakers_map[comps[7]]
                else:
                    speaker_id = speakers[0]
                    speakers.remove(speaker_id)
                    speakers_map[comps[7]] = speaker_id

                segments.append(Segment(speaker_id, begin_time, end_time))
            
        segments.sort(key=lambda seg: seg.start)
        with open(os.path.join(out_dir, talk + '.segments.json'), "w") as fp:
            json.dump([obj.__dict__ for obj in segments], fp)

def clean_audios():
    dir = "./vox_test/test_audio"
    out_dir = "./vox_test/audios"
    talks = [file.split(".")[0] for file in os.listdir(dir)]
    for talk in talks:
        with wave.open(os.path.join(dir, talk + ".wav"), 'rb') as audio:
            params = audio.getparams()
            sample_width = params.sampwidth
            frame_rate = params.framerate
            num_channels = params.nchannels
            total_frames = params.nframes

            frame_len = int(30 * frame_rate)
            cur_frame = 0
            cur_chunk = 0
            while cur_frame < total_frames:
                end_frame = min(cur_frame + frame_len, total_frames)

                with wave.open(os.path.join(out_dir, talk + "." + str(cur_chunk) + ".wav"), 'wb') as chunk_file:
                    # Set the parameters for the output WAV file
                    chunk_file.setparams((num_channels, sample_width, frame_rate, end_frame - cur_frame, params.comptype, params.compname))

                    # Read frames from the input WAV file and write to the output WAV file
                    audio.setpos(cur_frame)
                    chunk_file.writeframes(audio.readframes(end_frame - cur_frame))

                # Update the start frame and chunk index for the next iteration
                cur_frame = end_frame
                cur_chunk += 1

if __name__ == '__main__':
    # clean_labels()
    clean_audios()