from pyannote.metrics.diarization import DiarizationErrorRate, JaccardErrorRate
from pyannote.core import Annotation, Segment

class DiarizationReference():
    def __init__(self, reference_rttm):
        self.reference_rttm = reference_rttm
    
    # Loads RTTMs to Annotations for metric calculations
    def _load_rttm_to_annotation(self, rttm_file):
        annotation = Annotation()
        with open(rttm_file, 'r') as file:
            for line in file:
                parts = line.strip().split()

                start_time = float(parts[3])
                duration = float(parts[4])
                end_time = start_time + duration
                speaker = parts[7]
                segment = Segment(start_time, end_time)
                annotation[segment] = speaker
        return annotation

    # For a list of hypothesis RTTM files, calculate metrics for each and return
    # similarly indexed array.
    def calculate_der(self, hypotheses_rttms, type):
        if type == 'der':
            metric = DiarizationErrorRate()
        elif type == 'jer':
            metric = JaccardErrorRate()

        reference = self._load_rttm_to_annotation(self.reference_rttm)
        metrics = []
        for i in range(len(hypotheses_rttms)):
            hypothesis = self._load_rttm_to_annotation(hypotheses_rttms[i])
            metrics.append(metric(reference, hypothesis))

        return metrics
