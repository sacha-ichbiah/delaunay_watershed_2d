

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi

from skimage.segmentation import watershed
from skimage.feature import peak_local_max

def create_mesh_semantic_masks(Verts,Edges, image_shape): 
    Mask = np.zeros(image_shape)
    for edge in Edges : 
        v1,v2,_,_ = edge
        c = trace_line(Verts[v1],Verts[v2])
        Mask[tuple(c.T)]=1
    return(Mask)

def trace_line(v0, v1):
    x0, y0 = v0
    x1, y1 = v1
    "Bresenham's line algorithm - modified from https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#Python"
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append([x,y])
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append([x,y])
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy        
    points.append([x,y])
    return(np.array(points,dtype=np.int64))  

def create_mesh_instance_masks(Verts, Edges, image_shape, seeds):

    semantic_mask = create_mesh_semantic_masks(Verts,Edges, image_shape)
    distance = ndi.distance_transform_edt(1-semantic_mask)
    mask = np.zeros(distance.shape, dtype=bool)
    markers = np.zeros(distance.shape)
    for i in range(len(seeds)): 
        markers[tuple(seeds[i].T)]=i+1
    labels = watershed(-distance, markers)
    return(labels)


def reconstruct_mask_from_dict(filename_dict,plot = False): 
    Dict_mask = np.load(filename_dict,allow_pickle = True).item()
    Verts = Dict_mask["Verts"]
    Edges = Dict_mask["Edges"]
    seeds = Dict_mask["seeds"]
    image_shape = Dict_mask["image_shape"]
    labels = create_mesh_instance_masks(Verts,Edges, image_shape,seeds) -1
    if plot : 
        plt.figure(figsize = (4,4))
        plt.imshow(labels,plt.cm.nipy_spectral)
        plt.axis("off")
        plt.title("Reconstructed mask")
    return(labels)
   