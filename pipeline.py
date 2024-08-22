from conversion import mp3_to_wav, json_to_rttm, txt_to_rttm
from segmentation import AudioSegmenting, FileSegmenting
from embedding import Embedding
from clustering import ConstrainedAgglomerative, AgglomerativeCOPKmeans
from metrics import DiarizationReference

import torch
import numpy as np
import datetime


def time(secs):
    return datetime.timedelta(seconds=round(secs))

"""
Pipeline Deliverable
Given...
=> n_speakers: number of speakers in audio
=> audio_path: path to file to diarize
=> transcription_path: path to audio's reference transcription
=> output_transcription_path: path to transcription output (assume RTTM)
=> supervision_coeffs: array of coefficients, running the pipeline for each item providing the coefficient% of labeled segments
Generate RTTM for each supervision coefficient and hold DER and JER metrics for each.
"""
class Pipeline():

    def __init__(self, n_speakers, audio_path, transcription_path, output_transcription_path, supervision_coeffs=[0.2]):
        self.n_speakers = n_speakers
        self.audio_path = audio_path
        self.transcription_path = transcription_path
        self.output_transcription_path = output_transcription_path
        self.supervision_coeffs = supervision_coeffs
        self.metrics = () # for each supervision coeffs
        self.msg_id = 1

    def _print_progress(self, msg):
        print(f"[{self.msg_id}] " + msg)
        self.msg_id = self.msg_id + 1

    def _output_transcript_txt(self, output_path, labels, segments):
        assert(len(labels) == len(segments))
        f = open(output_path, "w")
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(int(labels[i] + 1))
        for (i, segment) in enumerate(segments):
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                f.write("\n" + segment["speaker"] + ' ' + str(time(segment["start"])) + '\n')
            
            f.write(segment["text"][1:] + ' ')
        f.close()  

    def run(self):
        self._print_progress(f"Initializing pipeline for {self.audio_path}.")
        """
            Phase 1: Conversion
        """
        wav_path = self.audio_path
        rttm_path = self.transcription_path

        if self.audio_path[-3:] == 'mp3':
            self._print_progress(f"Converting MP3 ({self.audio_path}) to WAV...")
            wav_path = self.audio_path.split('.')[0] + '.wav'
            mp3_to_wav(self.audio_path, wav_path)

        if self.transcription_path[-4:] == 'json':
            self._print_progress(f"Converting JSON ({self.transcription_path}) to RTTM...")
            rttm_path = self.transcription_path.split('.')[0] + '.rttm'
            json_to_rttm(self.transcription_path, rttm_path)
        
        """
            Phase 2: Segmentation
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._print_progress(f"Segmenting with {device}.")

        # Most expensive part of pipeline, no need to run for each supervision coefficient
        audio_segmenting = AudioSegmenting(device)
        audio_segmenting.transcribe(self.audio_path)
        unlabeled_segments = audio_segmenting.get_segments(proportion_coeff=1.0)
        self._print_progress(f"Created {len(unlabeled_segments)} unlabeled segments")

        file_segmenting = FileSegmenting(rttm_path)
        file_segmenting.transcribe()
        all_labeled_segments = file_segmenting.get_segments(proportion_coeff=1.0)
        self._print_progress(f"Created {len(all_labeled_segments)} labeled segments")

        """
            Phase 3: Embedding
        """
        embedding = Embedding(device, wav_path)
        self._print_progress(f"Embedding with {device}.")
        
        unlabeled_embeddings, _ = embedding.create_embeddings(unlabeled_segments)
        self._print_progress(f"Generated {len(unlabeled_embeddings)} unlabeled embeddings.")
        
        all_labeled_embeddings, all_true_labels = embedding.create_embeddings(all_labeled_segments)
        self._print_progress(f"Generated {len(all_labeled_embeddings)} labeled embeddings.")

        """
        Running pipeline for each given supervision coefficient
        """
        output_rttms = [] # list of all output rttm paths
        for supervision_coeff in self.supervision_coeffs:
            # Select subset of all labeled embeddings based on coefficient
            subset_size = int(supervision_coeff * len(all_labeled_embeddings))
            labeled_embeddings = all_labeled_embeddings[:subset_size]
            true_labels = all_true_labels[:subset_size]
            self._print_progress(f"Using first {int(supervision_coeff*100)}% ({len(labeled_embeddings)}) embeddings.")

            """
                Phase 4: Clustering
            """

            # Conversions betweeen given transcript's names and integer IDs
            speaker_name_to_label = {name: i for i, name in enumerate(np.unique(true_labels))}
            label_to_speaker_name = {v: k for k, v in speaker_name_to_label.items()}
            true_labels_int = np.array([speaker_name_to_label[name] for name in true_labels]) # integer labels

            combined_embeddings = np.vstack((labeled_embeddings, unlabeled_embeddings))
            clustering_model = ConstrainedAgglomerative(
                n_clusters=self.n_speakers,
                labeled_embeddings=labeled_embeddings,
                unlabeled_embeddings=unlabeled_embeddings,
                labels=true_labels_int
            )
            clustering_model.fit()
            cluster_labels = clustering_model.labels

            txt_output_name = self.output_transcription_path.split('.')[0] + str(supervision_coeff) + '.txt'
            rttm_output_name = self.output_transcription_path.split('.')[0] + str(supervision_coeff) + '.rttm'
            self._output_transcript_txt(txt_output_name, cluster_labels, unlabeled_segments)
            txt_to_rttm(txt_output_name, rttm_output_name)

            diarization_ref = DiarizationReference(rttm_path)
            output_rttms.append(rttm_output_name)
            self._print_progress(f"Generated RTTM ({rttm_output_name}).")
        der_scores = diarization_ref.calculate_metrics(output_rttms, 'der')
        jer_scores = diarization_ref.calculate_metrics(output_rttms, 'jer')
        self.metrics = (der_scores, jer_scores)
        self._print_progress(f"Calculated DER/JER metrics: {self.metrics}")

if __name__ == "__main__":
    # Usage example with 11.mp3
    audio_path = 'eleven/11.mp3'
    transcription_path = 'eleven/transcript.json'
    output_transcription_path = 'eleven/output.rttm'
    pipeline = Pipeline(
        n_speakers=9,
        audio_path=audio_path,
        transcription_path=transcription_path,
        output_transcription_path=output_transcription_path,
        supervision_coeffs=[0.1, 0.2, 0.4, 0.6, 0.8]
    )
    pipeline.run()