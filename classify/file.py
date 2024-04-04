import argparse
import sys
import time
import os
import cv2
from image_classifier import ImageClassifier
from image_classifier import ImageClassifierOptions

# Import GPIO and servo control functions
import RPi.GPIO as GPIO

# GPIO pin for servo control
SERVO_PIN = 18

# Set up GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(SERVO_PIN, GPIO.OUT)
servo = GPIO.PWM(SERVO_PIN, 50)  # PWM frequency = 50Hz

# Servo control function
def set_servo_angle(class_name):
    if class_name == 'soda_cans':
        angle = 0  # Move to 0 degrees for "soda_cans"
    elif class_name == 'water_bottle':
        angle = 90  # Move to 90 degrees for "water_bottle"
    else:
        angle = 50  # Move to 50 degrees for other classes

    duty = angle / 18 + 2
    GPIO.output(SERVO_PIN, True)
    servo.ChangeDutyCycle(duty)
    time.sleep(0.5)
    GPIO.output(SERVO_PIN, False)
    servo.ChangeDutyCycle(0)

_FPS_AVERAGE_FRAME_COUNT = 10


def run(model: str, max_results: int, num_threads: int, enable_edgetpu: bool,
        camera_id: int, width: int, height: int) -> None:
    """Continuously run inference on images acquired from the camera.
    Args:
        model: Name of the TFLite image classification model.
        max_results: Max of classification results.
        num_threads: Number of CPU threads to run the model.
        enable_edgetpu: Whether to run the model on EdgeTPU.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
    """

    # Initialize the image classification model
    options = ImageClassifierOptions(
        num_threads=num_threads,
        max_results=max_results,
        enable_edgetpu=enable_edgetpu)
    classifier = ImageClassifier(model, options)

    # Start the servo
    servo.start(0)

    while True:
        # Capture image from camera
        os.system("fswebcam /home/pi/Pictures/image.jpg")

        # Test image
        # image_path = "/home/pi/RVM/Lobe/tflite_model/can.jpg"
        image_path = "/home/pi/Pictures/image.jpg"
        image = cv2.imread(image_path)
        image = cv2.flip(image, 1)    # Optional preprocessing step

        # List classification results
        categories, final_prediction = classifier.classify(image)

        # Show classification results on the image
        for idx, category in enumerate(categories):
            class_name = category.label
            score = round(category.score, 2)
            result_text = class_name + ' (' + str(score) + ')'
            print("Prediction: {}, Probability: {}".format(class_name, str(score)))

        print("Final prediction: ", final_prediction)

        # Control the servo based on the final prediction
        set_servo_angle(final_prediction)

        # Allow time for the servo to move
        time.sleep(0.5)

          # Wait for a key press to exit
        if cv2.waitKey(1) == 27:
            break

    # Clean up GPIO
    servo.stop()
    GPIO.cleanup()


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
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
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
        bool(args.enableEdgeTPU), int(args.cameraId), args.frameWidth,
        args.frameHeight)


if __name__ == '__main__':
    main()
