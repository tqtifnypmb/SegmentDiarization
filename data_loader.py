import json
import torch
from ami_clean_data import Segment
import math
import os
from custom_whisper import log_mel_spectrogram, pad_or_trim, load_model

class DataLoader:
    def __init__(self, audios_dir, labels_dir) -> None:
        self.audios_dir = audios_dir
        self.labels_dir = labels_dir

        talks = [file for file in os.listdir(self.audios_dir)]
        talks.sort()
        
        self.train = talks[0:int(0.9 * len(talks))]
        self.validate = talks[int(0.9 * len(talks)):int(0.95 * len(talks))]
        self.test = talks[int(0.95 * len(talks)):]
        self.speakers = { speaker: index for index, speaker in enumerate(list("ABCDEF"))}

    @property
    def num_of_classes(self):
        return len(self.speakers) * (len(self.speakers) + 1) + 1

    def load_audio(self, file: str):
        url = os.path.join(self.audios_dir, file)
        audio = log_mel_spectrogram(url)
        audio = pad_or_trim(audio, 3000)
        assert audio.shape[1] == 3000
        return audio
    
    def load_labels(self, file: str):
        talk = file.split(".")[0]
        chunk = int(file.split(".")[1])
        seg_start = chunk * 30 * 1000
        seg_end = (chunk + 1) * 30 * 1000
        with open(os.path.join(self.labels_dir, talk + ".segments.json"), "r") as fp:
            labels = json.load(fp)
            segments = [Segment(**values) for values in labels]

        y = torch.zeros(1, 1500).to(torch.long)
        orders: list[str] = []
        validSegments: list[Segment] = []
        unique = set()
        for segment in segments:
            if int(segment.end * 1000) <= seg_start or int(segment.start * 1000) >= seg_end:
                continue
            
            beg = max(seg_start, int(segment.start * 1000))
            end = min(seg_end, int(segment.end * 1000))

            beg_index = int(math.ceil((beg - seg_start) / 20))
            if beg_index == 1500:
                continue

            if segment.speaker not in unique:
                unique.add(segment.speaker)
                orders.append(segment.speaker)

            validSegments.append(segment)

        if len(orders) > len(self.speakers):
            print(file)
            return None

        assert(len(orders) <= len(self.speakers))

        for segment in validSegments:
            beg = max(seg_start, int(segment.start * 1000))
            end = min(seg_end, int(segment.end * 1000))

            beg_index = int(math.ceil((beg - seg_start) / 20))
            end_index = int(math.ceil((end - seg_start) / 20))

            order = orders.index(segment.speaker)
            
            speaker_id = order * (len(self.speakers) + 1) + 1

            no_speakers = y == 0
            have_other_speakers = (y != 0).logical_and(y != speaker_id)
            one_speaker = have_other_speakers.logical_and(y % (len(self.speakers) + 1) == 1)
            
            if no_speakers[:, beg_index: end_index].nonzero().shape[0] > 0:
                mask = no_speakers
                mask[:, :beg_index] = False
                mask[:, end_index:] = False
                y[mask] = speaker_id

            if one_speaker[:, beg_index: end_index].nonzero().shape[0] > 0:
                mask = one_speaker
                mask[:, :beg_index] = False
                mask[:, end_index:] = False
                y[mask] += order + 1

        return y