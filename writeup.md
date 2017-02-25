#**Finding Lane Lines on the Road** 


### Reflection

###1. Describe pipeline.
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

One potential shortcoming would be cannot do average for short edges. Because short edges are almost horinoztal, I ignored these lines. But, these lines could be included.
Another shortcoming could be in hough-detected-edge. Maybe I would be wrong for hough-transform parameters.
Short lines are not detected by hough-transform.

###3. Suggest possible improvements to your pipeline

A possible improvement would be for averaging lines.
I guess the short line segements are not good to be averaged. 
If applying line fitting algorithmss, It would be better. 
Line or curve fitting is possible.

