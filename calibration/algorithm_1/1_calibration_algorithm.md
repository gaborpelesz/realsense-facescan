This text describes the first experimental RGB-D extrinsic calibration approach we tried. The approach uses a face as the calibration object.

This is a sort of naive approach.

The approach is the following:
1. the user should stand in front of the two cameras and his/her face should be approximately in the center of the image
1. init a list S to store for future transformations
1. **while** S has less than **N** elements **or** we reached **K** iterations:
    1. acquire two RGB-D images simultaneously from the two RGB-D cameras
    1. create a point cloud from each of them
    1. try colored ICP registration of the point clouds with init transformation as the last element if any
        - **if** the registration was successful **and** the error is below T1 threshold -> add the transformation to S
        - **if** the registration was unsuccessful -> report to user
1. the average of the results should be a good transformation