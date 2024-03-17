import cv2
import argparse
import os

from ultralytics import YOLO
import supervision as sv
import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 image detection")
    parser.add_argument("--image-path", type=str, required=True,
                        help="Path to the input image")
    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()

    # Load the input image
    image_path = args.image_path
    if not os.path.exists(image_path):
        print("Error: Image file does not exist.")
        return

    input_image = cv2.imread(image_path)

    # Load the YOLO model
    model = YOLO("best.pt")

    # Perform object detection on the input image
    result = model(input_image)


    # Convert YOLO result to Supervision format
    if isinstance(result, list):
        # If YOLOv8 result is a list of detections
        detections = sv.Detections.from_yolov8(result[0])
    else:
        # If YOLOv8 result is a single detection
        detections = sv.Detections.from_yolov8(result)

    # Annotate the input image with detected objects
    box_annotator = sv.BoxAnnotator(thickness=2, text_thickness=2, text_scale=1)
    annotated_image = box_annotator.annotate(scene=input_image, detections=detections)

    # Display the annotated image
    cv2.imshow("Detected Objects", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
