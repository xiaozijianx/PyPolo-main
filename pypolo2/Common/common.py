from re import A
import numpy as np

def even_dist(map_shape,  sigma = 5):
  dst = np.ones(map_shape)
  return dst / np.sum(dst)

def gaussian_dist(map_shape, sigma = 5, muu = 0.000):
  x, y = np.meshgrid(np.linspace(0,*map_shape), np.linspace(0, *map_shape))
  x_center = map_shape[0] // 2
  y_center = map_shape[1] // 2
  dst = np.sqrt((x - x_center)*(x - x_center)+(y - y_center)* (y - y_center))
  # Calculating Gaussian array
  gauss = np.exp(-( (dst-muu)**2 / ( 2.0 * sigma**2 ) ) )
  return gauss / np.sum(gauss)

def dg_corner_dist(map_shape,  sigma = 5, muu = 0.000):
  x, y = np.meshgrid(np.linspace(0,*map_shape), np.linspace(0, *map_shape))
  dst1 = np.sqrt((x)*(x)+(y)* (y))
  dst2 = np.sqrt((x - map_shape[0])*(x - map_shape[0])+(y - map_shape[1])* (y - map_shape[1]))
  # Initializing sigma and muu

  # Calculating Gaussian array
  gauss1 = np.exp(-( (dst1-muu)**2 / ( 2.0 * sigma**2 ) ) )
  gauss2 = np.exp(-( (dst2-muu)**2 / ( 2.0 * sigma**2 ) ) )
  return (gauss1 + gauss2) / np.sum(gauss1 + gauss2)

def noise_dist(map_shape):
  dist = np.abs(np.random.randn(*map_shape))
  return dist / np.sum(dist)

def GenerateEvenReqs(access_mask):
  return access_mask / np.sum(access_mask)

def GenerateGaussianReqs(access_mask, gaussian_dots=[[42, 4], [26, 2], [10, 10], [43, 26], [57, 7], [57, 25]], sigma=7):
  assert(access_mask.min() == 0)
  assert(access_mask.max() == 1)
  req_dist = np.zeros_like(access_mask).astype(np.float32)
  m_shape = access_mask.shape
  x, y = np.meshgrid(np.linspace(0, m_shape[0], m_shape[0]), np.linspace(0, m_shape[1], m_shape[1]))
  x = np.transpose(x)
  y = np.transpose(y)
  for dot in gaussian_dots:
    dst = np.sqrt((x - dot[0]) ** 2 + (y - dot[1]) ** 2)
    req_dist += np.exp(-( (dst)**2 / ( 2.0 * sigma**2 ) ))

  req_dist *= access_mask
  return req_dist / np.sum(req_dist)

def GenerateLaplacianReqs(access_mask, gaussian_dots=[[42, 4], [26, 2], [10, 10], [43, 26], [57, 7], [57, 25]], sigma=7):
  assert(access_mask.min() == 0)
  assert(access_mask.max() == 1)
  req_dist = np.zeros_like(access_mask).astype(np.float32)
  m_shape = access_mask.shape
  x, y = np.meshgrid(np.linspace(0, m_shape[0], m_shape[0]), np.linspace(0, m_shape[1], m_shape[1]))
  x = np.transpose(x)
  y = np.transpose(y)
  for dot in gaussian_dots:
    dst = np.sqrt((x - dot[0]) ** 2 + (y - dot[1]) ** 2)
    req_dist += np.exp(-( np.abs(dst) / ( sigma ) ))

  req_dist *= access_mask
  return req_dist / np.sum(req_dist)