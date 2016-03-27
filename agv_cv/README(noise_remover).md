Noise filter for lane detector's output
---------------------------------------
---------------------------------------

To run the code-

1.  g++ -ggdb `pkg-config --cflags opencv` -o `basename noise_remover.cpp .cpp` noise_remover.cpp `pkg-config --libs opencv`
2. ./noise_remover

Mat img =imread( "../images/noise.png", CV_LOAD_IMAGE_GRAYSCALE );
Change ^ to input image's location. The output image is saved as computer_vision/images/lanes.jpg.

----------------------------------------

How the code works-

The input image is filtered using median filter/ erosion & dilution. After this step, thick lines, big blogs and the lanes are left in the image. All the white pixels are stored in a vector 'points'. If 'points' is empty, there was probably some problem with input image and the program terminates with -1 return value.

Ransac(explained below) is used to find the best fitting quadratic equation for the co-ordinates in 'points'.The co-ordinates lying on this curve as marked and removed from 'points'. Ransac is again run on the remaining points to give the second lane. If the number of points on the second curve was below some threshold, the program concludes that input image has only one lane. Points from both the curves are marked on the image 'output' and is saved as computer_vision/images/lanes.jpg.

-------------------------------------------

Ransac(Random sample consensus)-

General ransac pseudocode-
1.Select a random subset of original data.
2.Model is fitted to the set.
3.All other data is tested against the fitted model. Points that fit within some error margin are considered as consensus set.
4.The model is reasonably good is consensus set is large, and if not repeat.

Three white points are randomly chosen, the quadratic curve through them is computed and the the number of white points on the curve is calculated. This process is repeated (~100 times) and the curve on which highest number of points lie is stored. This curve is the best fitting curve. Since we are randomly choosing 3 points, we may get a useless curve. But the probability of getting useless curves in every iteration is very low.

Suppose 30% of the points lie on first lane, 30% on second lane and remaining 40% of the points are noise.
P(detect 3 points from first lane)=0.4*0.4*0.4
P(not detecting all 3 points from first lane)=1-0.4*0.4*0.4=0.9360
P(not detecting all 3 points from first lane in 100 iterations)=0.936^100=0.0013
P(detecting the first lane atleast once)=1-0.0013=0.9987

Detecting the lane only once is enough because the lane would have the highest number of the points lying on the curve and hence will be marked as the best fit lane.

--------------------------------------------

Scope of improvement-

1. The threshold for second lane has to be adjusted.
2. Since the lanes are assumed to be quadratic curves, complete lanes will not be detected in case of sharp turns(>~90 deg).
3. If very large blobs are present near the lanes, correct lanes are not be detected. (computer_vision/images/noise2.png gives correct lanes but computer_vision/images/noise3.png doesn't)
