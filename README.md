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

## SIFT > (Scale Invariant Feature Transform)
### Algorithm
   
1.	Scale Space Construction
Search over multiple scales and image locations

![5](https://github.com/GhofranMohamed/CV_task3/assets/93389441/31289d80-ff12-4c2f-8e2b-588683091ffa)


2.	Scale Space Extrema Detection
●	Detect maxima and minima of differences of Gaussian in scale space.
●	Each point is compared to its 8 neighbours in the current image and 9 neighbours each in the
scales above and below.
●	Reject flats
●	The Hessian matrix was used to eliminate edge responses.

![6](https://github.com/GhofranMohamed/CV_task3/assets/93389441/28741de3-5156-4448-83b8-3a70d8d5a904)


4.	Orientation Assignment
●	Create histogram of local gradient directions at selected scale.
●	Assign canonical orientation at peak of smoothed histogram
●	Each key specifies stable 2D coordinates
●	Histogram of gradient orientation bin-counts
are weighted by gradient magnitudes and a Gaussian Weighting function. Usually,36 bins are chosen for the orientation.

𝑚(𝑥, 𝑦) = √(𝐿(𝑥 + 1, 𝑦) − 𝐿(𝑥 − 1, 𝑦))2 + (𝐿(𝑥, 𝑦 + 1) − 𝐿(𝑥, 𝑦 − 1))2
𝜃(𝑥, 𝑦) = tan−1((𝐿(𝑥, 𝑦 + 1) − 𝐿(𝑥, 𝑦 − 1))⁄(𝐿(𝑥 + 1, 𝑦) − 𝐿(𝑥 − 1, 𝑦)))


![7](https://github.com/GhofranMohamed/CV_task3/assets/93389441/607c0f25-bc30-4835-b81e-d45b668fac58)

4.	Key point descriptor
Use local image gradients at selected scale and rotation to describe each key point region

![8](https://github.com/GhofranMohamed/CV_task3/assets/93389441/c085056e-47c3-423c-9135-cec2962b6e30)


### Results 
Computation time equals 66.9212273999999 in seconds.
![9](https://github.com/GhofranMohamed/CV_task3/assets/93389441/0f03d89a-4e3e-48ad-997e-c6a18a49f309)


# Tab2:
Matching Harris corner points
## SSD
The aim is to first detect Harris corners from two images of the same scene. Then, image patches of size 15x15 pixels around each detected corner point is extracted following a matching step where mutually nearest neighbors are found using the sum of squared differences (SSD) similarity measure.
![10](https://github.com/GhofranMohamed/CV_task3/assets/93389441/9be9fa5b-69fa-41d2-ab5c-8135e13cf0ea)

### Algorithm 
•	Harris corner extraction
•	We pre-allocate the memory for the 15*15 image patches extracted around each corner point from both images, then extracts the patches using bilinear interpolation
•	We compute the sum of squared differences (SSD) of pixels' intensities for all pairs of patches extracted from the two images
•	Next we compute pairs of patches that are mutually nearest neighbors according to the SSD measure
•	We sort the mutually nearest neighbors based on the SSD
•	Estimate the geometric transformation between images
•	Next we visualize the 40 best matches which are mutual nearest neighbors and have the smallest SSD values
•	Finally, since we have estimated the planar projective transformation, we can check that how many of the nearest neighbor matches actually are correct correspondences

### Results 
![11](https://github.com/GhofranMohamed/CV_task3/assets/93389441/77aefd54-e9d8-4bf9-97c9-16e76acfa2ef)

## NCC 
from SSD to NCC:
o	You need to determine the mutually nearest neighbors by finding pairs for which NCC is maximized (i.e. not minimized like SSD).
o	Also, you need to sort the matches in descending order in terms of NCC
o	in order to find the best matches (i.e. not ascending order as with SSD).
![12](https://github.com/GhofranMohamed/CV_task3/assets/93389441/11123887-c6e8-415c-ae71-b99b9a4d5c71)

### Algorithm
•	Harris corner extraction
•	We pre-allocate the memory for the 15*15 image patches extracted around each corner point from both images, then extracts the patches using bilinear interpolation
•	We compute the sum of squared differences (SSD) of pixels' intensities for all pairs of patches extracted from the two images
•	Compute Normalized cross correlation for each window
•	Next we compute pairs of patches that are mutually nearest neighbors according to the ncc measure
•	We sort the mutually nearest neighbors based on the ncc
•	Estimate the geometric transformation between images
•	Next we visualize the 40 best matches which are mutual nearest neighbors and have the smallest ncc values
•	Finally, since we have estimated the planar projective transformation, we can check that how many of the nearest neighbor matches actually are correct correspondences

### Results 
![13](https://github.com/GhofranMohamed/CV_task3/assets/93389441/10b7ae1e-84ec-44f4-aa57-add66950673c)

### Comments 
1)	Using NCC, 81 correct correspondences were found compared to the 24 found with SSD.
2)	SDD is very sensitive to pixel intensity differences in images, so in this case, NCC works better because it normalizes the pixel intensities to account for the intensity difference between images. Also these two images only differs in translation and brightness, there is no significant rotation, scale or non-linear photometric differences, and NCC still works well under such circumstances

## Matching points using SIFT 
### Algorithm
•	Find the keypoints and descriptors with SIFT detector
•	Initiate BruteForce matcher with default params
•	Perform matching and save k=2 nearest neighbors for each descriptor
•	Apply Lowe's ratio test
•	Sort matches
•	Collect feature points and scales from the match objects
•	Estimate the geometric transformation between images

### Results 
![14](https://github.com/GhofranMohamed/CV_task3/assets/93389441/b0f8efc1-00b1-4cee-881c-f7e920921d7e)



