from landmark_detector import LandmarkDetector
import numpy as np

def main():
    detector = LandmarkDetector()
    landmarks, err = detector.detect('img_1.png', [819.27, 432.15, 971.70, 575.87])
    print(landmarks.round().astype(np.int))

if __name__ == '__main__':
  main()
