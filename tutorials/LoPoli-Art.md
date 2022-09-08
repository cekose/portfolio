---
layout: page
author: Cem Kose
title: It's all Polygons
---

# It's all Polygons

You ever do that exercise in art class way back when, where you would draw a bunch of small points and a few larger points and join them together using a straight edge? You’d end up with a brutal looking geometric image, then you’d colour it in to make it look groovy.

It’s a lot of fun. Weirdly calming… Anyway I recently discovered a programmatic approach to generating low-poly art that works on this principle.

![png]({{ site.url }}/assets/images/tutorials/project-1/geo.jpg){: width="100%" }

---

The approach is straight forward.

1. Load an image and process it to highlight distinct areas of detail.
2. Connect areas of interest with line segments.
3. Triangulate vertices to generate polygons.
4. Determine the colour of the triangles.
5. Compute and output a final image.

---

Let’s begin by importing our dependencies and loading our image.


```python
import os
import numpy as np
import pygame
import pygame.gfxdraw
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from scipy.ndimage import gaussian_filter
from collections import defaultdict
```

To highlight distinct areas in our image we’re going to apply “blurring” to the original sample.  We'll be using a Gaussian filter included in the scipy library to reduce the complexity of the image by removing smaller features.

We’ll be applying two Gaussian filters of varying intensity to our original sample to generate two blurred images. We will then measure the differences between them to pinpoint areas of importance to us.

Before we add a filter lets reduce the complexity by flattening the image into a 2D array by representing the image as a ratio of its RGB (Red, Green, Blue) pixels.


```python
input_image = pygame.surfarray.pixels3d(pygame.image.load("/home/fruit.png"))
RGB_weight = np.array([0.2126, 0.7152, 0.0722])
grayscale = (input_image * RGB_weight).sum(axis=-1)
plt.imshow(grayscale.T);
```


![png]({{ site.url }}/assets/images/tutorials/project-1/output_3_0.png)


We can then apply the Gaussian filters on the flattened image and then workout the difference between the two newly blurred images.


```python
blur = gaussian_filter(grayscale, 2, mode="reflect")
blur_x2 = gaussian_filter(grayscale, 30, mode="reflect")

# Take the difference, deweight negatives, normalise
difference = (blur - blur_x2)
difference[difference < 0] *= 0.1
difference = np.sqrt(np.abs(difference) / difference.max())

plt.imshow(difference.T)
```

![png]({{ site.url }}/assets/images/tutorials/project-1/output_5_1.png)



Now that we’ve found points of interest. We’ll  reduce the points further by picking at random x and y coordinates and eliminating points below a certain luminosity threshold.


```python
def sample(ref, n=1000000):
    np.random.seed(0)
    w, h = x.shape
    xs = np.random.randint(0, w, size=n)
    ys = np.random.randint(0, h, size=n)
    value = ref[xs, ys]
    accept = np.random.random(size=n) < value
    points = np.array([xs[accept], ys[accept]])
    return points.T, value[accept]
```


```python
samples, v = sample(difference)
plt.scatter(samples[:, 0], -samples[:, 1], c=v, s=0.2, edgecolors="none", cmap="viridis")
```

![png]({{ site.url }}/assets/images/tutorials/project-1/output_8_1.png)



To triangulate between vertices we’ll be using Delaunay triangulation included in the scipy library.

To determine the colour of a triangle we will assign each pixel to a triangle. The average of all the pixels within the triangle will determine the final colour of the triangle.


```python
def get_colour_of_tri(tri, image):
    colours = defaultdict(lambda: [])
    w, h, _ = image.shape
    for i in range(0, w):
        for j in range(0, h):
            # Gets the index of the triangle the point is in
            index = tri.find_simplex((i, j))
            colours[int(index)].append(inp[i, j, :])
    # For each triangle, find the average colour
    for index, array in colours.items():
        colours[index] = np.array(array).mean(axis=0)
    return colours
```


```python
def draw(tri, colours, screen, upscale):
    s = screen.copy()
    for key, c in colours.items():
        t = tri.points[tri.simplices[key]]
        pygame.gfxdraw.filled_polygon(s, t * upscale, c)
        pygame.gfxdraw.polygon(s, t * upscale, c)
    return s
```

That should be it. Let’s plot the final image.


```python
w, h, _ = inp.shape
upscale = 2
screen = pygame.Surface((w * upscale, h * upscale))
screen.fill(inp.mean(axis=(0, 1)))
corners = np.array([(0, 0), (0, h - 1), (w - 1, 0), (w - 1, h - 1)])
points = np.concatenate((corners, samples))

outdir = "lowpoly/output/"
os.makedirs(outdir, exist_ok=True)

for i in range(0, 100):
    n = 5 + i + 2 * int(i**2)
    tri = Delaunay(points[:n, :])
    colours = get_colour_of_tri(tri, inp)
    s = draw(tri, colours, screen, upscale)
    s = pygame.transform.smoothscale(s, (w, h))
    pygame.image.save(s, f"lo-poli/output/{i:04d}.png")
```
