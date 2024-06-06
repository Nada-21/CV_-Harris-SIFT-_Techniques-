import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.measure import ransac
from scipy.ndimage.interpolation import map_coordinates
from harris import*
from SIFT import*
import time 
import logging
logger = logging.getLogger(__name__)
from Functions import circle_points

#...........................Sum of Squared Differences..........................................................
def match_SSD (I1,I2):
  start = time.time()
  # Harris corner extraction
  x1, y1, image1 = harris(I1)
  x2, y2, image2 = harris(I2)
  # We pre-allocate the memory for the 15*15 image patches extracted
  # around each corner point from both images
  patch_size=15
  npts1=x1.shape[0]
  npts2=x2.shape[0]
  patches1=np.zeros((patch_size, patch_size, npts1))
  patches2=np.zeros((patch_size, patch_size, npts2))
  # The following part extracts the patches using bilinear interpolation
  k=(patch_size-1)/2.
  xv,yv=np.meshgrid(np.arange(-k,k+1),np.arange(-k, k+1))
  for i in range(npts1):
    patch = map_coordinates(I1, (yv + y1[i], xv + x1[i]))
    patches1[:,:,i] = patch
  for i in range(npts2):
    patch = map_coordinates(I2, (yv + y2[i], xv + x2[i]))
    patches2[:,:,i] = patch
  # We compute the sum of squared differences (SSD) of pixels' intensities
  # for all pairs of patches extracted from the two images
  distmat = np.zeros((npts1, npts2))
  for i1 in range(npts1):
      for i2 in range(npts2):
          distmat[i1,i2]=np.sum((patches1[:,:,i1]-patches2[:,:,i2])**2)
  # Next we compute pairs of patches that are mutually nearest neighbors
  # according to the SSD measure
  ss1 = np.amin(distmat, axis=1)
  ids1 = np.argmin(distmat, axis=1)
  ss2 = np.amin(distmat, axis=0)
  ids2 = np.argmin(distmat, axis=0)
  pairs = []
  for k in range(npts1):
      if k == ids2[ids1[k]]:
          pairs.append(np.array([k, ids1[k], ss1[k]]))
  pairs = np.array(pairs)

  # We sort the mutually nearest neighbors based on the SSD
  sorted_ssd = np.sort(pairs[:,2], axis=0)
  id_ssd = np.argsort(pairs[:,2], axis=0)

  # Estimate the geometric transformation between images
  src=[]
  dst=[]
  for k in range(len(id_ssd)):
      l = id_ssd[k]
      src.append([x1[int(pairs[l, 0])], y1[int(pairs[l, 0])]])
      dst.append([x2[int(pairs[l, 1])], y2[int(pairs[l, 1])]])
  src=np.array(src)
  dst=np.array(dst)
  rthrs=2
  tform,_ = ransac((src, dst), ProjectiveTransform, min_samples=4,
                residual_threshold=rthrs, max_trials=1000)
  H1to2p = tform.params

  # Next we visualize the 40 best matches which are mutual nearest neighbors
  # and have the smallest SSD values
  Nvis = 40

  montage = np.concatenate((I1, I2), axis=1)

  fig = plt.figure()
  plt.imshow(montage, cmap='gray')
  plt.axis('off')
  
  for k in range(np.minimum(len(id_ssd), Nvis)):
      l = id_ssd[k]
      plt.plot(x1[int(pairs[l, 0])], y1[int(pairs[l, 0])], 'rx')
      plt.plot(x2[int(pairs[l, 1])] + I1.shape[1], y2[int(pairs[l, 1])], 'rx')
      plt.plot([x1[int(pairs[l, 0])], x2[int(pairs[l, 1])]+I1.shape[1]], 
         [y1[int(pairs[l, 0])], y2[int(pairs[l, 1])]])
      
  p1to2=np.dot(H1to2p, np.hstack((src, np.ones((src.shape[0],1)))).T)
  p1to2 = p1to2[:2,:] / p1to2[2,:]
  p1to2 = p1to2.T
  pdiff=np.sqrt(np.sum((dst-p1to2)**2, axis=1))

  n_correct = len(pdiff[pdiff<rthrs])

  # Ouput the execution time
  exe_time = str(time.time() - start)

  return fig, n_correct, exe_time

#....................................Normalized Cross Correlation...........................................................
def match_NCC (I1,I2):

  start = time.time()
  # Harris corner extraction
  x1, y1, cimg1 = harris(I1)
  x2, y2, cimg2 = harris(I2)
  # According to SSD Function
  patch_size=15
  npts1=x1.shape[0]
  npts2=x2.shape[0]
  patches1=np.zeros((patch_size, patch_size, npts1))
  patches2=np.zeros((patch_size, patch_size, npts2))
  distmat = np.zeros((npts1, npts2))
  # The following part extracts the patches using bilinear interpolation
  k=(patch_size-1)/2.
  xv,yv=np.meshgrid(np.arange(-k,k+1),np.arange(-k, k+1))

  for i in range(npts1):
    patch = map_coordinates(I1, (yv + y1[i], xv + x1[i]))
    patches1[:,:,i] = patch

  for i in range(npts2):
    patch = map_coordinates(I2, (yv + y2[i], xv + x2[i]))
    patches2[:,:,i] = patch

  for i1 in range(npts1):
    for i2 in range(npts2):
      distmat[i1,i2]=np.sum((patches1[:,:,i1]-patches2[:,:,i2])**2)

  ss1 = np.amin(distmat, axis=1)
  # Compute Normalized cross correlation for each windows
  ncc = np.zeros((npts1, npts2))
  for i1 in range(npts1):
    for i2 in range(npts2):
      n1 = patches1[:,:,i1] - np.mean(patches1[:,:,i1])
      n2 = patches2[:,:,i2] - np.mean(patches2[:,:,i2])
      ncc[i1,i2] = np.sum(n1*n2)/np.sqrt(np.sum(n1**2)*np.sum(n2**2))
  # Next we compute pairs of patches that are mutually nearest neighbors
  # according to the ncc measure
  ncc1 = np.amax(ncc, axis=1)
  ids1 = np.argmax(ncc, axis=1)
  ncc2 = np.amax(ncc, axis=0)
  ids2 = np.argmax(ncc, axis=0)

  pairs = []
  for k in range(npts1):
    if k == ids2[ids1[k]]:
      pairs.append(np.array([k, ids1[k], ss1[k]]))
  pairs = np.array(pairs)

  # We sort the mutually nearest neighbors based on the ncc
  sorted_ncc = np.sort(pairs[:,2], axis=0)[::-1]
  id_ncc = np.argsort(pairs[:,2], axis=0)[::-1]
  # Estimate the geometric transformation between images
  src=[]
  dst=[]
  for k in range(len(id_ncc)):
    l = id_ncc[k]
    src.append([x1[int(pairs[l, 0])], y1[int(pairs[l, 0])]])
    dst.append([x2[int(pairs[l, 1])], y2[int(pairs[l, 1])]])

  src=np.array(src)
  dst=np.array(dst)
  rthrs=2
  tform,_ = ransac((src, dst), ProjectiveTransform, min_samples=4,
        residual_threshold=rthrs, max_trials=1000)
  H1to2p = tform.params
  # Next we visualize the 40 best matches which are mutual nearest neighbors
  # and have the smallest ncc values
  Nvis = 40

  montage = np.concatenate((I1, I2), axis=1)

  fig = plt.figure()
  plt.imshow(montage, cmap='gray')
  plt.axis('off')

  for k in range(np.maximum(len(id_ncc), Nvis)):
      l = id_ncc[k]
      plt.plot(x1[int(pairs[l, 0])], y1[int(pairs[l, 0])], 'rx')
      plt.plot(x2[int(pairs[l, 1])] + I1.shape[1], y2[int(pairs[l, 1])], 'rx')
      plt.plot([x1[int(pairs[l, 0])], x2[int(pairs[l, 1])]+I1.shape[1]], 
           [y1[int(pairs[l, 0])], y2[int(pairs[l, 1])]])
  # Finally, since we have estimated the planar projective transformation
  # we can check that how many of the nearest neighbor matches actually
  # are correct correspondences
  p1to2=np.dot(H1to2p, np.hstack((src, np.ones((src.shape[0],1)))).T)
  p1to2 = p1to2[:2,:] / p1to2[2,:]
  p1to2 = p1to2.T
  pdiff=np.sqrt(np.sum((dst-p1to2)**2, axis=1))
  # The criterion for the match being a correct is that its correspondence in
  # the second image should be at most rthrs=2 pixels away from the transformed
  n_correct = len(pdiff[pdiff<rthrs])

  # Ouput the execution time
  exe_time = str(time.time() - start)

  return fig, n_correct, exe_time

#..........................Matching Using SIFT....................................
def match_sift(img1,img2):

  start = time.time()
  #Get keypoints, descriptors
  kp1, des1= computeKeypointsAndDescriptors(img1)
  kp2, des2= computeKeypointsAndDescriptors(img2)
  # Initiate BruteForce matcher with default params
  bf = cv2.BFMatcher()
  # Perform matching and save k=2 nearest neighbors for each descriptor
  matches = bf.knnMatch(des1, des2, k=2)
  # Apply Lowe's ratio test
  good_matches = []
  for m,n in matches:
    if m.distance < 0.75*n.distance:
      good_matches.append(m)
  # Sort matches 
  good_matches = sorted(good_matches, key = lambda x:x.distance)
  # Collect feature points and scales from the match objects
  source_pts = []
  target_pts = []
  source_radii = []
  target_radii = []

  for match in good_matches:
    # Collect feature point coords and scale query (img1)
    x, y = kp1[match.queryIdx].pt
    pt = np.array([np.round(x), np.round(y)]).astype(np.int)
    source_pts.append(pt)
    radius = kp1[match.queryIdx].size / 2.
    source_radii.append(radius)

    # Collect feature point coords and scale query (img2)
    x, y = kp2[match.trainIdx].pt
    pt = np.array([np.round(x), np.round(y)]).astype(np.int)
    target_pts.append(pt)
    radius = kp2[match.trainIdx].size / 2.
    target_radii.append(radius)

  source_pts = np.array(source_pts)
  source_radii = np.array(source_radii)
  target_pts = np.array(target_pts)
  target_radii = np.array(target_radii)

  ## Estimate the geometric transformation between images
  rthrs=10
  tform,_ = ransac((source_pts, target_pts), SimilarityTransform, min_samples=2,
                                 residual_threshold=rthrs, max_trials=1000)
  H1to2p = tform.params
  s = np.sqrt(np.linalg.det(H1to2p[0:2,0:2]))
  R = H1to2p[0:2,0:2] / s
  t = H1to2p[0:2,2]

  montage = np.concatenate((img1, img2), axis=1)
  Nvis = 20
  fig=plt.figure()
  plt.title("Matching points using SIFT")
  plt.imshow(montage, cmap='gray')
  plt.axis("off")

  for k in range(0, Nvis):   
      plt.plot([source_pts[k,0], target_pts[k,0]+img1.shape[1]],\
               [source_pts[k,1], target_pts[k,1]], 'r-')

      x,y=circle_points(source_pts[k,0], source_pts[k,1],\
                        3*np.sqrt(2)*source_radii[k])
      plt.plot(x, y, 'r', linewidth=1.5)

      x,y=circle_points(target_pts[k,0]+img1.shape[1], target_pts[k,1],\
                        3*np.sqrt(2)*target_radii[k])
      plt.plot(x, y, 'r', linewidth=1.5)


  # Ouput the execution time
  exe_time = str(time.time() - start)
 
  return fig, exe_time
