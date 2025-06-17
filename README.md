# Compare-pictures-and-videos
Combines several images/videos into one (for comparison purposes).

On this repository, there are 3 images examples you can use for a test:
- "1. Image 1.jpg"
- "2. Image 2.jpg"
- "3. Image 3.jpg"

By double clicking on **run.bat**, you'll get a final image named **combined_image.jpg** that'll give you this result:

![combined_image](https://github.com/user-attachments/assets/d2ddfeae-d2c7-4c2b-8f37-f9d9977715f0)

It's also working for videos (and videos + images):

![ezgif-20e4452739f7de-FINAL](https://github.com/user-attachments/assets/aa8f5096-abd6-4158-9757-d550b9894af6)

You can change the title [here](https://github.com/BigStationW/Compare-pictures-and-videos/blob/6f66c25a5fb9b9575c2e92b21a75297d41b7a8d6/combine.py#L10)


## 1. Clone the repository somewhere on your PC
```git clone https://github.com/BigStationW/Compare-pictures-and-videos```


## 2. Install the required Python packages
```pip install -r requirements.txt```

## 3. Your images must be on the same folder as [the script](https://github.com/BigStationW/Compare-pictures/blob/main/combine.py) and the run.bat file
- Naming the files as "1. X.jpg", "2. Y.jpg" and so on tells the program the order in which the images should appear from left to right.
- The images can be in any of the following formats: ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
- The videos can be in any of those following formats: ['.mp4', '.avi', '.mov', '.mkv', '.webm']

