import cv2
import numpy as np
import json
import os

width = 640


def rotate_image_90_degrees(image):
    rotated_image = [[0 for _ in range(width)] for _ in range(width)]
    for i in range(width):
        for j in range(width):
            rotated_image[i][j] = image[width - j - 1][i]
    return np.array(rotated_image)


def flip_vertical(image):
    flipped_matrix = []
    for i in range(width - 1, -1, -1):
        flipped_matrix.append(image[i])
    return np.array(flipped_matrix)


def flip_horizontal(image):
    flipped_matrix = []
    for i in range(width):
        flipped_matrix.append(image[i][::-1])
    return np.array(flipped_matrix)


def get_files_in_directory(directory_path):
    os.chdir(directory_path)
    file_names = os.listdir()
    return file_names


def convert_bbox2seg(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[0],
        bbox[1] + bbox[2] - 1,
        bbox[0] + bbox[3] - 1,
        bbox[1],
        bbox[0] + bbox[3] - 1,
        bbox[1] + bbox[2] - 1,
    ]


def seg_rot90(A):
    output = []
    for i in range(0, len(A), 2):
        output.append(A[i + 1])
        output.append(width - A[i] + 1)
    return output


def seg_flip_h(A):
    output = []
    for i in range(0, len(A), 2):
        output.append(A[i])
        output.append(width - A[i + 1] + 1)
    return output


def seg_flip_v(A):
    output = []
    for i in range(0, len(A), 2):
        output.append(width - A[i] + 1)
        output.append(A[i + 1])
    return output


def convert_bbox2seg(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[0],
        bbox[1] + bbox[2] - 1,
        bbox[0] + bbox[3] - 1,
        bbox[1],
        bbox[0] + bbox[3] - 1,
        bbox[1] + bbox[2] - 1,
    ]


def anot_bbox(bbox, type):
    A = convert_bbox2seg(bbox)
    if type == "90":
        A = seg_rot90(A)
    elif type == "270":
        for i in range(3):
            A = seg_rot90(A)
    elif type == "v":
        A = seg_flip_v(A)
    elif type == "h":
        A = seg_flip_h(A)
    elif type == "vh":
        A = seg_flip_v(A)
        A = seg_flip_h(A)
    elif type == "90h":
        A = seg_rot90(A)
        A = seg_flip_h(A)
    elif type == "90v":
        A = seg_rot90(A)
        A = seg_flip_v(A)
    small_value = A[0] ** 2 + A[1] ** 2
    small_index = 0
    big_value = A[0] ** 2 + A[1] ** 2
    big_index = 0
    for i in range(1, 4):
        i_value = A[i * 2] ** 2 + A[i * 2 + 1] ** 2
        if small_value > i_value:
            small_value = i_value
            small_index = i
        if big_value < i_value:
            big_value = i_value
            big_index = i
    return [
        A[small_index * 2],
        A[small_index * 2 + 1],
        A[big_index * 2 + 1] - A[small_index * 2 + 1] + 1,
        A[big_index * 2] - A[small_index * 2] + 1,
    ]


json_path = "C:/Users/mamad/OneDrive/Desktop/final-project/augmentation/test/_annotations.coco.json"
directory_path = (
    "C:/Users/mamad/OneDrive/Desktop/final-project/augmentation/test/"
)
file_names = get_files_in_directory(directory_path)
for i in file_names:
    if i[-3:] == "jpg":
        path = directory_path + i
        original_image = cv2.imread(path)
        rotated_90 = rotate_image_90_degrees(original_image)
        rotated_270 = original_image
        for j in range(3):
            rotated_270 = rotate_image_90_degrees(rotated_270)
        fliped_v = flip_vertical(original_image)
        fliped_h = flip_horizontal(original_image)
        fliped_hv = flip_vertical(fliped_h)
        rotated_90_fliped_h = flip_horizontal(rotated_90)
        rotated_90_fliped_v = flip_vertical(rotated_90)
        name = path[:-4]
        cv2.imwrite(name + "_rotated_90.jpg", rotated_90)
        cv2.imwrite(name + "_rotated_270.jpg", rotated_270)
        cv2.imwrite(name + "_fliped_v.jpg", fliped_v)
        cv2.imwrite(name + "_fliped_h.jpg", fliped_h)
        cv2.imwrite(name + "_fliped_hv.jpg", fliped_hv)
        cv2.imwrite(name + "_rotated_90_fliped_h.jpg", rotated_90_fliped_h)
        cv2.imwrite(name + "_rotated_90_fliped_v.jpg", rotated_90_fliped_v)
print("Complete generate pictures...")

with open(
    json_path,
    "r",
) as file:
    json_data = json.load(file)

last_id_annotations = int(json_data["annotations"][-1]["id"])
last_id_images = int(json_data["images"][-1]["id"])
images_num = last_id_images + 1
annotations_num = last_id_annotations + 1
name_list = [
    "_rotated_90.jpg",
    "_rotated_270.jpg",
    "_fliped_v.jpg",
    "_fliped_h.jpg",
    "_fliped_hv.jpg",
    "_rotated_90_fliped_h.jpg",
    "_rotated_90_fliped_v.jpg",
]
image_id_converted = {}
for i in range(images_num):
    data_images = json_data["images"][i]
    name = data_images["file_name"][:-4]
    date_captured = data_images["date_captured"]
    license = data_images["license"]
    for j in range(7):
        x_id = last_id_images + j + 1
        id_str = str(i) + "_" + str(j)
        image_id_converted.update({id_str: x_id})
        new_data = {
            "id": x_id,
            "license": license,
            "file_name": name + name_list[j],
            "height": width,
            "width": width,
            "date_captured": date_captured,
        }
        with open(
            json_path,
            "r+",
        ) as file:
            json_data = json.load(file)
            json_data["images"].append(new_data)
            file.seek(0)
            json.dump(json_data, file, indent=4)
    last_id_images += 7
print("Complete add images info...")

for i in range(annotations_num):
    data_annotations = json_data["annotations"][i]
    image_id = data_annotations["image_id"]
    category_id = data_annotations["category_id"]
    bbox = data_annotations["bbox"]
    area = data_annotations["area"]
    segmentation = data_annotations["segmentation"]
    iscrowd = data_annotations["iscrowd"]
    for j in range(7):
        b_box = []
        segment = segmentation[0]
        if j == 0:
            b_box = anot_bbox(bbox, "270")
            for k in range(3):
                segment = seg_rot90(segment)
        elif j == 1:
            b_box = anot_bbox(bbox, "90")
            segment = seg_rot90(segment)
        elif j == 2:
            b_box = anot_bbox(bbox, "h")
            segment = seg_flip_h(segment)
        elif j == 3:
            b_box = anot_bbox(bbox, "v")
            segment = seg_flip_v(segment)
        elif j == 4:
            b_box = anot_bbox(bbox, "vh")
            segment = seg_flip_v(segment)
            segment = seg_flip_h(segment)
        elif j == 5:
            b_box = anot_bbox(bbox, "90h")
            segment = seg_rot90(segment)
            segment = seg_flip_h(segment)
        else:
            b_box = anot_bbox(bbox, "90v")
            segment = seg_rot90(segment)
            segment = seg_flip_v(segment)
        id_find = str(image_id) + "_" + str(j)
        new_data = {
            "id": last_id_annotations + j + 1,
            "image_id": image_id_converted[id_find],
            "category_id": category_id,
            "bbox": b_box,
            "area": area,
            "segmentation": segment,
            "iscrowd": iscrowd,
        }
        with open(
            json_path,
            "r+",
        ) as file:
            json_data = json.load(file)
            json_data["annotations"].append(new_data)
            file.seek(0)
            json.dump(json_data, file, indent=4)
    last_id_annotations += 7
print("Complete add annotations info...")
