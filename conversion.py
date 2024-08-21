import json
import subprocess
import shutil
import re


def json_to_rttm(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    with open(output_path, 'w') as f_o:
        for segment in data['ep-11']:
            start = segment['utterance_start']
            duration = segment['duration']
            speaker = segment['speaker']
            line = f"SPEAKER 11 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker.replace(' ', '_')} <NA> <NA>\n"
            f_o.write(line)

def mp3_to_wav(mp3_path, output_path):
    if shutil.which("ffmpeg") is not None:
        subprocess.call(['ffmpeg', '-i', mp3_path, '-ac', '1', output_path, '-y'])
    else:
        print("ffmpeg is not installed or not found in the system's PATH.")

# Convert time in H:M:S format to seconds.
def time_to_seconds(time_str):
    h, m, s = map(float, time_str.split(':'))
    return h * 3600 + m * 60 + s

def txt_to_rttm(input_txt, output_rttm):
    with open(input_txt, 'r') as file:
        lines = file.readlines()

    previous_start = -1
    previous_speaker = ""
    with open(output_rttm, 'w') as rttm_file:
        for line in lines:
            if line.strip():  # Skip empty lines
                # Match lines like "SPEAKER 3 0:00:00"
                match = re.match(r"SPEAKER (\d+) (\d+:\d+:\d+)", line.strip())
                if match:
                    speaker = match.group(1)
                    start_time = match.group(2)
                    start_seconds = time_to_seconds(start_time)
                    if previous_start == -1:
                        previous_start = start_seconds
                        previous_speaker = speaker
                    else:
                        duration = start_seconds - previous_start
                        line = f"SPEAKER 11 1 {previous_start:.3f} {duration:.3f} <NA> <NA> {previous_speaker.replace(' ', '_')} <NA> <NA>\n"
                        rttm_file.write(line)
                        previous_start = start_seconds
                        previous_speaker = speaker
