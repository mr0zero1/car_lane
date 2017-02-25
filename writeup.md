#**Finding Lane Lines on the Road** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file. But feel free to use some other method and submit a pdf if you prefer.

---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.
My pipeline is as follows:
 1. convert image to gray scale 
 2. get canny image
 3. get hough transformed edges 
 4. filter out invalid lines as follows: 
     - Remove short invalid lines because extrapolation is wrong for these for lines. 
       (ex. short line to extrapolates horinoztal lines)
 5. average lines
     - To average lines, all cartesian space lines converted to hough(theta,rho) space point.
     - average in hough space
 6. Extrapolate lines
     - convert hough space point (averaged) into cartesian space.
 7. draw lines
     - extrapolated lines as red line.
     - Hough edges as cyan lines.
 


###2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when ... 

Another shortcoming could be ...


###3. Suggest possible improvements to your pipeline

A possible improvement would be to ...

Another potential improvement could be to ...
