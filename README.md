# Description:
A small web application based app developed with python and streamlit, to apply different image processing techniques.
# Requirements:
•	Python 3.
•	Streamlit 1.13.0
•	Numpy 1.23.4
•	Matplotlib 3.6.2

# Running command:
Streamlit run server.py

o	The UI contains two main tabs Features Generation, Matching

# Tab1:

## Harris:
Harris Corner Detector is a corner detection operator that is commonly used in computer vision algorithms to extract corners and infer features of an image. Corners are the important features in the image, and they are generally termed as interest points which are invariant to translation, rotation, and illumination. Harris, and Stephens developed the Harris Corner Detector [1], a mathematical approach to detect corners and edges in images. They picked the statements of Moravec and gave it a mathematical signification, Equation 1.

![1](https://github.com/GhofranMohamed/CV_task3/assets/93389441/ae51fb6f-3a9e-4549-ad92-6a7718592bf0)
After applying Taylor expansion it is possible to obtain the following approximation of Equation2.
![2](https://github.com/GhofranMohamed/CV_task3/assets/93389441/c118868e-5e93-4421-b12a-3373b7793968)

The Ix and Iy present in Equation 2 are the x and y image derivatives. To conclude, the response of the corner detector is obtained with Equation 3. Depending on the value obtained by this response, is possible to determine if a region contains a flat region, an edge, or a corner.
![3](https://github.com/GhofranMohamed/CV_task3/assets/93389441/a6ff31b2-8dc4-478c-af7e-44fc069a2b0d)

### Algorithm
in this algorithm we change the image type to float then we do the next steps
1.	Calculate image x and y derivatives.
2.	Derivate again the previous values to obtain the second derivative;
3.	For each pixel, sum the last step obtained derivatives. Here we are making a 1 pixel sift of the windows over the image;
4.	For each pixel and using the sums of the previous step, define H matrix;
5.	Calculate the response of the detector;
6.	Use a threshold value in order to exclude some of the detections.
   
### Results 
Computation time equals 0.11477828025817871 seconds on average
![4](https://github.com/GhofranMohamed/CV_task3/assets/93389441/cc5c6a52-4c33-4714-8739-a2bad534b1d8)

## SIFT 
Scale Invariant Feature Transform, used to extract features from images using 4 steps:              ![5](https://github.com/GhofranMohamed/CV_task3/assets/93389441/31289d80-ff12-4c2f-8e2b-588683091ffa)                   
1.	Scale space construction
2.	Scale space extrema detection
3.	Orientation Assignment
4.	Key point descriptor
1.	Scale Space Construction
Search over multiple scales and image locations
