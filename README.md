# Binary-Image-Segmentation
Binary Image segmentation is the process of classifying the pixels of an image into two categories: pixels belonging to the foreground objects of an image and pixels belonging to the background objects of an image. Figure 1 shows an example of image segmentation where the pixels belonging to the foreground objects are highlighted in red and the pixels for the background are highlighted in blue. (Image I taken from the web). Image segmentation is an important problem in image processing and computer vision with many application ranging from background substraction and removal to object tracking, etc. While there are many ways to model this problem mathematically, the one we will use for this homework in as a min-cut finding problem with multiple sources and sinks [1] as described below. In this project, you are asked to implement a simple binary image segmentation technique using min-cut and the OpenCV library.

The program has 3 arguments: an input image, a configuration file that provides the initial set of foreground and background points and an output image. Your program has to compute the binary segmentation of the pixels in the input image based on the initial segmentation provided in the configuration file. The output has to be an image of the same size of the input image such that each pixel is either white (foreground) or black (background).


Prerequisites:

cmake, opencv


To compile in the lab
——————————

run ./lab_config.sh
cd build
make

To compile on your own machine
———

cd build/
cmake ../
make


To run
——

In the build folder:
./seg image1.jpg config.txt image2.jpg

 