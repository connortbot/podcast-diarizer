import conversion
from segmentation import AudioSegmenting, FileSegmenting
from embedding import Embedding
from clustering import ConstrainedAgglomerative, AgglomerativeCOPKmeans


"""
Pipeline Deliverable
=> audio_path: path to file to diarize
=> transcription_path: path to audio's reference transcription
=> output_transcription_path: path to transcription output (assume RTTM)
"""
class Pipeline():
    def __init__(self, audio_path, transcription_path, output_transcription_path):
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        self.output_transcription_path = output_transcription_path
    
    def run(self):
        # Phase 1: Conversion
        if self.audio_path[:-3] == 'mp3':
            pass # make wav conversion
        if self.transcription_path[:-4] == 'json':
            pass # convert to 'transcript.rttm'




if __name__ == "__main__":
    pipeline = Pipeline()
    pipeline.run()