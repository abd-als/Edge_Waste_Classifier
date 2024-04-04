import argparse
import sys
import time
import os
import cv2
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions

# Visualization parameters
_ROW_SIZE = 20  # pixels
_LEFT_MARGIN = 24  # pixels
_TEXT_COLOR = (0, 0, 255)  # red
_FONT_SIZE = 1
_FONT_THICKNESS = 1
_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, max_results: int, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
 
  # Initialize the image classification model
  options = ImageClassifierOptions(
      num_threads=num_threads,
      max_results=max_results,
      enable_edgetpu=enable_edgetpu)
  classifier = ImageClassifier(model, options)

  # Variables to calculate FPS
  counter, fps = 0, 0
  start_time = time.time()

  # Open the camera
  cap = cv2.VideoCapture(camera_id)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

  while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    if not ret:
      break

    # Flip the frame horizontally (optional preprocessing step)
    frame = cv2.flip(frame, 1)

    # List classification results
    categories, final_prediction = classifier.classify(frame)

    # Show classification results on the image
    for idx, category in enumerate(categories):
      class_name = category.label
      score = round(category.score, 2)
      result_text = class_name + ' (' + str(score) + ')'
      print("Prediction: {}, Probability: {}".format(class_name, str(score)))
  
    print("Final prediction: ", final_prediction)

    # Calculate the FPS
    if counter % _FPS_AVERAGE_FRAME_COUNT == 0:
      end_time = time.time()
      fps = _FPS_AVERAGE_FRAME_COUNT / (end_time - start_time)
      start_time = time.time()

    # Show the FPS
    fps_text = 'FPS = ' + str(int(fps)) + ' second(s)'
    print(fps_text)

    # Display the image
    cv2.imshow('frame', frame)

    # Wait for a key press to exit
    if cv2.waitKey(1) ==27:
      break

  # Release the camera and close the window
  cap.release()
  cv2.destroyAllWindows()


def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model',
      help='Name of image classification model.',
      required=False,
      default='modelPort.tflite')
  parser.add_argument(
      '--maxResults',
      help='Max of classification results.',
      required=False,
      default=3)
  parser.add_argument(
      '--numThreads',
      help='Number of CPU threads to run the model.',
      required=False,
      default=4)
  parser.add_argument(
      '--cameraId', help='Id of camera.', required=False, default=0)
  parser.add_argument(
      '--frameWidth',
      help='Width of frame to capture from camera.',
      required=False,
      default=640)
  parser.add_argument(
      '--frameHeight',
      help='Height of frame to capture from camera.',
      required=False,
      default=480)
  args = parser.parse_args()

  run(args.model, int(args.maxResults), int(args.numThreads),
      False, int(args.cameraId), int(args.frameWidth), int(args.frameHeight))

if __name__ == '__main__':
  main()
