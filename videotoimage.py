import cv2
import os
import numpy as np


def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


last_folder = ""
prev_count = 0
file_paths = os.listdir("./Images")
file_paths.sort()
count = 0
# print("fresh_onion", " -> ", len(os.listdir(f"./Images/fresh_onion")))

# for images in os.listdir(f"./Images/fresh_onion"):
#     if count % 2 == 0:
#         os.remove(f"./Images/fresh_onion/{images}")
#     count += 1

# print("fresh_onion", " -> ", len(os.listdir(f"./Images/fresh_onion")))

for file_path in file_paths:

    #     # for folder in os.listdir(f'./Images/{file_path}'):
    if os.path.isdir(f"./Images/{file_path}"):
        print(file_path, " -> ", len(os.listdir(f"./Images/{file_path}")))
#     if file_path.find(".mp4") == -1:
#         continue
#     vidcap = cv2.VideoCapture(f"./Videos/{file_path}")
#     success, image = vidcap.read()
#     folder_name = "_".join(file_path.split(".")[0].split("_")[:-1])
#     count = prev_count if last_folder == folder_name else 0
#     if not os.path.exists(f"./Images/{folder_name}"):
#         os.mkdir(f"./Images/{folder_name}")
#     while success:
#         image = cv2.resize(image, (224, 224))
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_0.jpg", image)
#         n_image = rotate_image(image, 45)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_45.jpg", n_image)
#         n_image = rotate_image(image, 90)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_90.jpg", n_image)
#         n_image = rotate_image(image, 135)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_135.jpg", n_image)
#         n_image = rotate_image(image, 180)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_180.jpg", n_image)
#         n_image = rotate_image(image, -45)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_-45.jpg", n_image)
#         n_image = rotate_image(image, -90)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_-90.jpg", n_image)
#         n_image = rotate_image(image, -135)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_-135.jpg", n_image)
#         v_flip = cv2.flip(image, 0)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_v_flip.jpg", v_flip)
#         h_flip = cv2.flip(image, 1)
#         cv2.imwrite(f"./Images/{folder_name}/frame_{count}_h_flip.jpg", h_flip)
#         success, image = vidcap.read()
#         count += 1
#     print(f"Total frames : {file_path} is {count * 7}")
#     last_folder = folder_name
#     prev_count = count
#     print(last_folder)
#     print(count)
