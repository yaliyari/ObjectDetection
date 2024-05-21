import os
image_files = ['apple.jpg', 'apples.jpg', 'car.jpg', 'car1.jpg', 'car2.jpg', 'car3.jpg', 'clock.jpg', 'clock2.jpg',
'clock3.jpg', 'fruits.jpg', 'oranges.jpg']

# output folder
def create_output_folder():
    dir_name = "image_with_box"
    if not os.path.exists(dir_name):
        print("output folder has been created")
        os.mkdir(dir_name)
    else:
        print("output folder existed!")