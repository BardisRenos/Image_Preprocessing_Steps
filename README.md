# Image_Preprocessing

<p align="center"> 
<img src="https://github.com/BardisRenos/Image_Preprocessing/blob/main/OPEN_CV.png" width="450" height="150" style=centerme>
</p>

In this repository will demostrates the ability of **Python** programming language and the library of **OpenCV** to apply process to images. 

Digital image processing is the use of computer algorithms to perform image processing on digital images. As a subfield of digital signal processing, digital image processing has many advantages over analogue image processing. It allows a much wider range of algorithms to be applied to the input data — the aim of digital image processing is to improve the image data (features) by suppressing unwanted distortions and/or enhancement of some important image features so that our AI-Computer Vision models can benefit from this improved data to work on.

Image processing is performed in three steps:

First, import images with an optical devices like a scanner or a camera, or make them by computer-generated imagery.

Second, manipulate or analyze the images in some way. This step can include image improvement and data summary, or the images are analyzed to find rules that aren't seen by the human eyes. For example, meteorologists use this processing to analyze satellite photographs.

Last, output the result of image processing. The result might be the image changed by some way or it might be a report based on analysis or result of the images.

### 1. Read the image ### 


```python
  img = cv2.imread('/home/eAdmin/Desktop/Images/100128_d6_front.png')
```
```python
  print(img.shape)
```

```
(958, 1276, 3)
```
In order to show the image that we read. I create a method to show each image for further developing.  

```python
def show_image(img):
    cv2.imshow("Given Image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
```

<p align="center"> 
<img src="https://github.com/BardisRenos/Image_Preprocessing/blob/main/Screenshot%20from%202020-12-07%2014-43-39.png" width="700" height="500" style=centerme>
</p>


The structure of an image is an array of 2D or 3D dimension. When is color image the array is 3D and black white then 2D.

```python
  print(img)
```

```
[[[0 0 0]
  [0 0 0]
  [0 0 0]
  ...
  [0 0 0]
  [0 0 0]
  [0 0 0]]

 [[0 0 0]
  [0 0 0]
  [0 0 0]
```

An image is nothing more than a two-dimensional array of numbers(or pixels) ranging between 0 and 255. It is defined by the mathematical function f(x,y) where x and y are the two co-ordinates horizontally and vertically.


### 2.Convert the color of the images ###
Convert an image from RGB structure to grayscale format. 

```python
cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
```
<p align="center"> 
<img src="https://github.com/BardisRenos/Image_Preprocessing/blob/main/Screenshot%20from%202020-12-07%2014-45-52.png" width="700" height="500" style=centerme>
</p>

### 3. Image Resizing ###
An image can be resized into our dimensions

```python
img_gray = cv2.resize(img_gray, (350, 350), interpolation=cv2.INTER_AREA)   
```
After the resize of the image. The picture is 350 by 350 pixels and one channel only. The indication that the image if in black white format.
```
(350, 350)
```

### 4. Noise ####
Managing images is a frequent problem is the noise. There are different kind of imaging noise. Here, we can give an overview of three basic types of noise that are common in image processing applications:

1.  Gaussian noise
2.  Random noise
3.  Salt and Pepper noise (Impulse noise – only white pixels)


The example code adds Salt and Pepper noise.

```python
def add_noise(img):
    from skimage.util import random_noise
    # Adding Gaussian noise
    noise_img = random_noise(img, mode='s&p', amount=0.3)
    noise_img = np.array(255*noise_img, dtype='uint8')
    show_image(noise_img)
```

#### 4.1 Gaussian Filter ####
```python

def gaussian_filter(img):
    img = add_noise(img)
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    show_multiple_images(img, gaussian, "Input Image", "Gaussian Filter")
```

#### 4.2 Median Filter ####

```python
def median_blur(img):
    img = add_noise(img)
    median = cv2.medianBlur(img, 5)
    show_multiple_images(img, median, "Input Image", "Median Filter")
```

#### 4.3 Bilateral Filter ####

```python
def bilateral(img):
    img = add_noise(img)
    blur = cv2.bilateralFilter(img, 9, 75, 75)
    show_multiple_images(img, blur, "Input Image", "Bilateral Filter")
```

### 5. Canny Edge Detection ###

```python
def canny_edge(img):
    canny_edge_image = cv2.Canny(img, 100, 200)
    show_2_images(img, canny_edge_image, "Input Image", "Edge Image")
```

<p align="center"> 
<img src="https://github.com/BardisRenos/Image_Preprocessing_Steps/blob/main/Screenshot%20from%202021-03-17%2015-32-26.png" width="900" height="500" style=centerme>
</p>

### 6. Segmentation & Morphology ###

```python
def segmentation_morphology(image, thresh):
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    show_multiple_images(image, opening, sure_bg, sure_fg, unknown, 'Original', 'Opening', "sure_bg", "sure_fg",
                         "The difference")
```

### 7. Image Gradients ###

An image gradient is a directional change in the intensity or color in an image. The gradient of the image is one of the fundamental building blocks in image processing.


### 8. Contours ###

Contours can be explained simply as a curve that joins all the continuous points (along the boundary), having same color or intensity. The contours are a useful tool for shape analysis and object detection and recognition.

```python
def contours_of_image(image, image_gray):
    thresh = threshold(image_gray)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # Draw all contours from the image
    # The number -1 signifies drawing all contours
    cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
    show_image(image)
```

<img src="https://github.com/BardisRenos/Image_Preprocessing_Steps/blob/main/Figure_1.png" width="300"/> <img src="https://github.com/BardisRenos/Image_Preprocessing_Steps/blob/main/Figure_2.png" width="300"/>

### 9. Image Segmentation with Watershed Algorithm ###

