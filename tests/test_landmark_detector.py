import pytest
from landmark_detection import LandmarkDetector

def test_landmark_detector():
    detector = LandmarkDetector()
    landmarks, err = detector.detect(
        'img_1.png',
        [819.27, 432.15, 971.70, 575.87]
    )
    assert err == ""