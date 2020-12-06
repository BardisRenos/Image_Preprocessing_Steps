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
vnfgnfvf

#### 4.1 Gaussian ####

#### 4.2 Median ####

### 5. Canny Edge Detection ###

### 6. Segmentation & Morphology ###

### 7. Image Gradients ###

### 8. Contours ###

### 9. Image Segmentation with Watershed Algorithm ###

