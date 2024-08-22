import whisper

class Segmenting:
    def __init__(self):
        self.segments = []

    def get_segments(self, proportion_coeff):
        subset_size = int(proportion_coeff * len(self.segments))
        subset_segments = self.segments[:subset_size]
        return subset_segments

class AudioSegmenting(Segmenting):
    def __init__(self, device):
        self.model = whisper.load_model("base").to(device)
        super().__init__()

    def transcribe(self, audio_path):
        result = self.model.transcribe(audio_path)
        self.segments = result['segments']

# Provides segments for given RTTM file
class FileSegmenting(Segmenting):
    def __init__(self, rttm_path):
        self.rttm_path = rttm_path
        super().__init__()
    
    def transcribe(self):
        labeled_segments = []
        with open(self.rttm_path, "r") as file:
            for line in file:
                parts = line.strip().split()
                start = float(parts[3])
                duration = float(parts[4])
                end = start + duration
                speaker = parts[7]
                labeled_segments.append({"start": start, "end": end, "text": "", "speaker": speaker})
        self.segments = labeled_segments
