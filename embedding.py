import torch
import wave
import numpy as np
from pyannote.audio import Audio
from pyannote.core import Segment
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding


class Embedding():
    def __init__(self, device, audio_path):
        # Note that output shape of model is (192,...)
        self.embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device(device))
        
        self.audio = Audio(sample_rate=16000, mono="downmix")
        self.audio_path = audio_path
        
    def _calculate_audio_duration(self):
        with wave.open(self.audio_path, 'r') as f:
            duration = f.getnframes() / float(f.getframerate())
        return duration
    
    def _create_segment_embedding(self, segment):
        start = segment['start']
        # True end of the segment, in case Whisper creates a segment beyond the duration.
        end = min(self._calculate_audio_duration(), segment['end'])
        block = Segment(start, end)
        waveform, _ = self.audio.crop(self.audio_path, block)

        return self.embedding_model(waveform[None])

    # Exposed embedding creator
    # => If embedding_labels has a "speaker" field, then will also return embedding_labels. If not,
    # => embedding labels will be []
    # returns embeddings, embedding_labels
    def create_embeddings(self, segments):
        embeddings = np.zeros(shape=(len(segments), 192))
        embedding_labels = []
        for i in range(len(segments)):
            embeddings[i] = self._create_segment_embedding(segments[i])
            if ('speaker' in segments[i]):
                embedding_labels.append(segments[i]['speaker'])
        embeddings = np.nan_to_num(embeddings)
        return embeddings, embedding_labels
