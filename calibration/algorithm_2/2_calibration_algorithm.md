# Using the Cubes - as accurate as possible

The naive algorithm's had poor results.

As I don't think that the problem of extrinsic calibration (relative pose) can be solved with ICP and depth, I will create a this algorithm to acquire a calibration with high accuracy.

I will do the extrinsic calibration on the RGB only.

Regarding the two-view geometry from the book (Multiple view geometry), I know the camera matrices P, P'. For accurate point correspondences I will use the famous cube from MIRA.

The calibration will be based on what I will read in the book. I will implement every part of the process.

At the end, I will evaluate the results compared to the other algorithm I will also see how accurately the calibration shows on my face point cloud.

## Steps of the algorithm

1. acquire two RGB images of a cube
2. manually label the edges of the cubes on the two image