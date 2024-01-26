import xml.etree.ElementTree as ET
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
    speakers = list("ABCDEFGHIJKLMN")
    talks = [file.split(".")[0] for file in os.listdir("./segments")]
    files = list(set(talks))
    for talk in files:
        segments = []
        for speaker in speakers:
            filename = "./segments/" + talk + "." + speaker + ".segments.xml"
            if not os.path.exists(filename):
                break

            doc = ET.parse(filename)
            for segment in doc.findall("segment"):
                segments.append(Segment(speaker, float(segment.attrib["transcriber_start"]), float(segment.attrib["transcriber_end"])))
        segments.sort(key=lambda seg: seg.start)
        with open("./labels/" + talk + '.segments.json', "w") as fp:
            json.dump([obj.__dict__ for obj in segments], fp)

def clean_audios():
    talks = [file for file in os.listdir("./Array1-01")]
    for talk in talks:
        if talk == ".DS_Store":
            continue
        if talk == "LICENCE.txt":
            continue

        with wave.open("./Array1-01/" + talk + "/audio/" + talk + ".Array1-01.wav", 'rb') as audio:
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

                with wave.open("./audios/" + talk + "." + str(cur_chunk) + ".wav", 'wb') as chunk_file:
                    # Set the parameters for the output WAV file
                    chunk_file.setparams((num_channels, sample_width, frame_rate, end_frame - cur_frame, params.comptype, params.compname))

                    # Read frames from the input WAV file and write to the output WAV file
                    audio.setpos(cur_frame)
                    chunk_file.writeframes(audio.readframes(end_frame - cur_frame))

                # Update the start frame and chunk index for the next iteration
                cur_frame = end_frame
                cur_chunk += 1

if __name__ == '__main__':
    clean_labels()
    # clean_audios()