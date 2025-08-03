import os

for root, dirs, files in os.walk('.'):
    for file in files:
        if 'haarcascade_frontalface_default.xml' in file:
            print("Found at:", os.path.join(root, file))
