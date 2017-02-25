#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
#%matplotlib inline

from scipy.optimize import curve_fit
import numpy as np

def f(x, A, B): # this is your 'straight line' y=f(x)
    return A*x + B

import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1,y1,x2,y2 in line:
            #print(">>>",x1,y1,x2,y2)
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)



def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


def degree(theta):
  return theta*180.0/np.pi

def is_left_line(line):
  [theta,rho, x1 ,y1 ,x2 , y2] = line
  #if ( 90 > degree(theta) ) : return 1
  if (x1 < 400): return 1
  if (x2 < 400): return 1
  #if ( 90 > degree(theta) ) : return 1
  return 0

def is_right_line(line):
  [theta,rho, x1 ,y1 ,x2 , y2] = line
  if (x1 > 500): return 1
  if (x2 > 500): return 1
 # if ( 90 < degree(theta) ) : return 1
  return 0

def hough_point(line):
  [x1,y1,x2,y2] = [float(line[0]),float(line[1]), float(line[2]), float(line[3]) ]
  # @tony 2017-02-23 : why? np.pi/2 is added ?
  # Because theta is perpendicular to the line. 
  # Please refer to the page : https://en.wikipedia.org/wiki/Hough_transform
  theta = np.arctan2(y2-y1, x2-x1) + np.pi/2.0;
  rho = (x2*y1 - y2*x1) / np.sqrt( (y2-y1)*(y2-y1) + (x2-x1)*(x2-x1) )
  return [theta, rho, x1, y1, x2, y2]

def valid_line(lines):
  return filter(is_valid_line,lines)  

def get_l_lines(lines):
  return filter(is_left_line, map(hough_point,lines))
  
def get_r_lines(lines):
  return filter(is_right_line,map(hough_point,lines))


def is_valid_line(line):
  [theta,rho, x1 ,y1 ,x2 , y2] = line
  # remove horiznotal line
  if np.sqrt( (x2-x1)*(x2-x1) + (y2-y1)*(y2-y1) ) < 30 :  return 0
  return 1


def flat_map(lines):
  result = []
  for line in lines:
    for x1,y1,x2,y2 in line:
      result.append([x1,y1,x2,y2])
  return result

def average_line(comment,lines):
    result = []

    if 0 == len(lines): return [[]]

    sum_theta = 0.0
    sum_rho   = 0.0
    count = 0
    for theta, rho, x1, y1, x2, y2 in lines:
      #print(comment,">>>", theta, theta*180/np.pi, rho ,x1, y1, x2, y2)
      sum_theta += theta
      sum_rho += rho

    average_theta = sum_theta / len(lines)
    average_rho   = sum_rho   / len(lines)

    #print(comment,"+++", average_theta,average_theta*180/np.pi , average_rho )
    return [[ cartesian_line(average_theta, average_rho) ]]


def fit_line(comment,lines):
    result = []

    if 0 == len(lines): return []
    x = []
    y = []
    for theta, rho, x1, y1, x2, y2 in lines:
      x.append(x1)
      x.append(x2)
      y.append(y1)
      y.append(y2)

    A,B = curve_fit(f, x, y)[0] # your data x, y to fit
    #print(comment,"+++", average_theta,average_theta*180/np.pi , average_rho )
    return [[ eval_line(A,B) ]]



def original_lines(comment,lines):
    result = []
    for theta, rho, x1, x2, y1, y2 in lines:
      result.append([ cartesian_line(theta, rho) ])
    return result
 
def cartesian_line(theta,rho):
  [x1, y1] = cartesian_point( theta, rho, 320 )
  [x2, y2] = cartesian_point( theta, rho, 1000 )
  return [ int(x1), int(y1), int(x2), int(y2) ]


def eval_line(A,B):
 # y = A*x + B
#  x = (y-B)/A
  [x1, y1] = eval_point( A, B, 320 )
  [x2, y2] = eval_point( A, B, 1000 )
  return [ int(x1), int(y1), int(x2), int(y2) ]


def eval_point(A,B,y):
  return [(y-B)/A, y ]

def cartesian_point(theta, rho, y):
  x = ( rho - y*np.sin(theta) ) / np.cos(theta) 
  return [x, y]


def average_hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    #  draw_lines(line_img, lines)
    l_lines =  average_line("l", valid_line(get_l_lines (flat_map(lines))))
    r_lines =  average_line("r", valid_line(get_r_lines (flat_map(lines))))
    draw_lines(line_img, old_if_no_l_line(l_lines), color=[255, 0, 0], thickness=10)
    draw_lines(line_img, old_if_no_r_line(r_lines), color=[255, 0, 0], thickness=10)
    draw_lines(line_img, lines, color=[0,255, 255], thickness=2)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, alpha=0.8, beta=1., lamda=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * alpha + img * beta + lamda
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, alpha, img, beta, lamda)


import os
os.listdir("test_images/")






# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
#plt.show()

#reading in an image

def test_pipeline(initial_image):
  low_threshold = 50
  high_threshold = 150
  kernel_size = 5
  full_edge_image = canny(gaussian_blur(grayscale(initial_image),kernel_size),low_threshold, high_threshold)


  imshape = initial_image.shape
  masked_area_poly = np.array([[(0,imshape[0]),(450, 350), (490, 350), (imshape[1],imshape[0])]], dtype=np.int32)
  masked_edge_image = region_of_interest(full_edge_image, masked_area_poly)

  # Define the Hough transform parameters
  # Make a blank the same size as our image to draw on
  rho = 2 # distance resolution in pixels of the Hough grid
  theta = np.pi/180 # angular resolution in radians of the Hough grid
  threshold = 15     # minimum number of votes (intersections in Hough grid cell)
  min_line_length = 20 #minimum number of pixels making up a line
  max_line_gap = 10   # maximum gap in pixels between connectable line segments
  line_image = average_hough_lines(masked_edge_image, rho, theta, threshold, min_line_length, max_line_gap) 
  color_edges = np.dstack((masked_edge_image, masked_edge_image, masked_edge_image)) 
  full_edge_image_3 = np.dstack((full_edge_image, full_edge_image, full_edge_image)) 

  overlapped_edge = weighted_img(color_edges, initial_image, 0.8, 1, 0)
  overlapped_image = weighted_img(line_image, overlapped_edge, 0.8, 1, 0)
  return overlapped_image


#initial_image = mpimg.imread('test_images/solidWhiteRight.jpg')



count = 0
def process_image(image):
  global count
#  mpimg.imsave("./output/org_image"+str(count),image)    
  result = test_pipeline(image)
#  mpimg.imsave("./output/out_image"+str(count), result)    
  count+=1  
  #print(">>>", count)
  return result

import os
#for file in os.listdir("test_images/"):
#  initial_image = mpimg.imread("test_images/"+file)
#  print('This image is:', type(initial_image), 'with dimensions:', initial_image.shape)
#  plt.imshow(test_pipeline(initial_image))  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
#  plt.show()

if 0:
  initial_image = mpimg.imread("./output/"+"org_image21.png")
  #initial_image = mpimg.imread("./test_images/"+"solidYellowLeft.jpg")
  print('This image is:', type(initial_image), 'with dimensions:', initial_image.shape)
  initial_image=np.asarray(initial_image[:,:,:3]*255, dtype="uint8")
  #print(">>>>>>>>",initial_image)
  plt.imshow(test_pipeline(initial_image))  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')
  plt.show()


def conv_video(clip1,white_output):
  clip1 = VideoFileClip("solidYellowLeft.mp4")
  white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
  white_clip.write_videofile(white_output, audio=False)

if 1:
  conv_video(VideoFileClip("solidWhiteRight.mp4"), 'white.mp4')
  conv_video(VideoFileClip("solidYellowLeft.mp4"), 'yellow.mp4')

