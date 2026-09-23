import Levenshtein
import difflib

def char_overlap_rate(pred: str, truth: str) -> float:  
    pred = str(pred) if pred is not None else ""
    truth = str(truth) if truth is not None else ""
    if len(pred) == 0 and len(truth) == 0:
        return 1.0
    if max(len(pred), len(truth)) == 0:
        return 0.0
    dist = Levenshtein.distance(pred, truth)
    return max(0.0, 1 - dist / max(len(pred), len(truth)))