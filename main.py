import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from sample_images import images_list


def show_sample_images():
    # os.chdir('./sample_images')
    names = images_list.image_files
    for image in names:
        print(f"\nDisplaying image: {image}")
        img = cv2.imread(f"./sample_images/{image}")
        cv2.imshow(f"{image}", img)
        cv2.waitKey(10)
    return names


def detect_and_draw_box(filename, model="yolov4", confidence=0.5):
    """This method detects objects and put a box around them
    Inputs:
    1- File name of the image, 2- model to be used, either yolov4 or yolov4_tiny (which is faster), 3- confidence level
    """
    # reading image
    img = cv2.imread(f"./sample_images/{filename}")

    # perform object detection
    bbox, label, conf = cv.detect_common_objects(img, confidence=confidence, model=model)

    # print image name:
    print(f"===========\nImage processed {filename}\n")

    # print detected objects
    for i, j in zip(label, conf):
        print(f"Detected object {i} with confidence level of {j} \n")


def main():
    images_list.create_output_folder()
    image_names = show_sample_images()
    for image in image_names:
        detect_and_draw_box(image)

if __name__ == "__main__":
    main()
