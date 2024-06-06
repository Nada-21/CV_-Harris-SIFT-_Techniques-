import numpy as np

# 2d Gaussian filter
def gaussian2(sigma, N=None):
 
    if N is None:
        N = 2*np.maximum(4, np.ceil(6*sigma))+1

    k = (N - 1) / 2.
            
    xv, yv = np.meshgrid(np.arange(-k, k+1), np.arange(-k, k+1))
    
    # 2D gaussian filter    
    g = 1/(2 * np.pi * sigma**2) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))

    # 1st order derivatives
    gx = -xv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma ** 2))
    gy = -yv / (2 * np.pi * sigma**4) * np.exp(-(xv**2 + yv**2) / (2 * sigma**2)) 

    # 2nd order derivatives
    gxx = (-1 + xv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gyy = (-1 + yv**2 / sigma**2) * np.exp(-(xv**2 + yv**2) / (2*sigma**2)) / (2 * np.pi * sigma**4)
    gxy = (xv * yv) / (2 * np.pi * sigma**6) * np.exp(-(xv**2 + yv**2) / (2*sigma**2))    

    return g, gx, gy, gxx, gyy, gxy

def maxinterp(v):
    a = 0.5*v[0] - v[1] + 0.5*v[2]
    b = -0.5*v[0] + 0.5*v[2]
    c = v[1]
    
    loc = (-b/2.0/a) 
    m = np.dot(np.array([a,b,c]), np.array([loc**2, loc, 1]))
    return m, loc

def circle_points(cx, cy, rad):
    # Calculate circle points for each keypoint
    theta = np.arange(0, 2*np.pi+0.1, 0.1)
    x = cx + np.cos(theta) * rad
    y = cy + np.sin(theta) * rad

    return x, y