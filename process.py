from PIL import Image
import os

for path in [
    "./VegetableDataSet/Fresh/brinjal",
    "./VegetableDataSet/Fresh/cauliflower",
    "./VegetableDataSet/Fresh/potato",
    "./VegetableDataSet/Fresh/tomato",
]:
    print(path)
    for i in range(-180, 180, 30):
        for file in os.listdir(path):
            image = Image.open(path + "/" + file)
            rotate_img = image.rotate(i)
            rotate_img.save(
                "./Converted/Fresh/"
                + path.split("/")[-1]
                + "/"
                + file.split(".")[0]
                + f"_{i}."
                + file.split(".")[1]
            )
