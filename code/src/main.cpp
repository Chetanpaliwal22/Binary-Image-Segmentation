//Project 1 
//References:
//https://stackoverflow.com/questions/11071509/opencv-convert-scalar-to-float-or-double-type
//https://www.geeksforgeeks.org/flood-fill-algorithm-implement-fill-paint/
//https://docs.opencv.org/2.4.13.7/doc/tutorials/imgproc/imgtrans/sobel_derivatives/sobel_derivatives.html

#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

class Edge
{
public:
  int source;
  int sink;
  int weight;
  bool fromSource;
  bool fromSink;


    Edge (int weight, int source, int sink, bool fromSource, bool fromsink)
  {
    weight = weight;
    fromSource = fromSource;
    fromSink = fromSink;
    source = source;
    sink = sink;
  }
};

class Vertex
{
public:
  void addEdge (Edge E)
  {
    edges.push_back (E);
  }
  vector < Edge > edges;
  int vertex;
  int parentVertex;
  bool source;
  bool sink;
  bool visited;

  Vertex (int vertex, int parentVertex, bool source, bool sink, bool visited)
  {
    vertex = vertex;
    parentVertex = parentVertex;
    source = source;
    sink = sink;
    visited = visited;
  }
};

void
addEdge (vector < pair < int, int > >adj[], int u, int v, int weight)
{
  adj[u].push_back (make_pair (v, weight));
  adj[v].push_back (make_pair (u, weight));
}

Mat finalImg;
int adjacencyMatrix[1000][500];
int v[1000][500];

bool breadth_first_search(vector<Vertex> &adjacencyList, int source,int sink){

	queue<Vertex> queue,edge;
	while(!queue.empty()){
		queue.pop();
	}
	Vertex node_to_push = adjacencyList.at(source);
	queue.push(node_to_push);
	while(!queue.empty()){
		Vertex current_node = queue.front();
		queue.pop();
		vector<Edge> edges;
		for(int i=0;i<edges.size();++i){
			//Edge edge;
			//edge = edges.at(i);
			/*Vertex adjacent_node = adjacencyList.at(edge.getToVertex());
			if(edge.getWeight() > 0 && !adjacent_node.isIsVisited()){
				adjacencyList.at(edge.getToVertex()).setIsVisited(true);
				adjacencyList.at(edge.getToVertex()).setParentVertex(current_node.getVertex());
				node_to_push = adjacencyList.at(edge.getToVertex());
				if(adjacent_node.isIsSink()){
					swap(queue,edge);
					return true;
				}
				queue.push(node_to_push);
			}*/
			}
		}
	return true;
}

void flood(int x, int y, int M, int N)
{
	// Base cases
	if (x < 0 || x >= N || y < 0 || y >= M || adjacencyMatrix[y][x]==0 || v[y][x]==1)
    	return;
    Vec3b p;
    p[0] = p[1] = p[2]= 255;
    v[y][x] = 1;
    finalImg.at < Vec3b > (y, x) = p;
	// Recur for north, east, south and west
	flood(x+1, y, M, N);
	flood(x-1, y, M, N);
	flood(x, y+1,  M, N);
	flood(x, y-1,  M, N);
}





int
main (int argc, char **argv)
{
  if (argc != 4)
    {
      cout << "Usage: ../seg input_image initialization_file output_mask" <<
	endl;
      return -1;
    }

  // Load the input image
  // the image should be a 3 channel image by default but we will double check that in teh seam_carving
  Mat in_image;
  in_image = imread (argv[1] /*, CV_LOAD_IMAGE_COLOR */ );

  if (!in_image.data)
    {
      cout << "Input Image not accessible. Not ablet ot load." << endl;
      return -1;
    }

  if (in_image.channels () != 3)
    {
      cout << "Image does not have 3 channels! " << in_image.depth () << endl;
      return -1;
    }

Mat sourceImage = in_image.clone();
  // the output image
  Mat out_image = in_image.clone ();

  ifstream f (argv[2]);
  if (!f)
    {
      cout << "Not able to load the config file." << endl;
      return -1;
    }

  int width = in_image.cols;
  int height = in_image.rows;
  int numberPixels = height * width;
  
  Mat gradient_image;

  GaussianBlur (sourceImage, sourceImage, Size (3, 3), 0, 0, BORDER_DEFAULT);
Mat gradient;
Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Scharr( sourceImage, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_x, abs_grad_x );
    Scharr( sourceImage, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT );
    convertScaleAbs( grad_y, abs_grad_y );
    addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, gradient );


cvtColor (gradient, gradient_image, COLOR_BGR2GRAY);
//normalize(gradient_image,gradient_image,0,255, NORM_MINMAX,CV_8UC1);

//Adjacency Matrix

//vector<Vertex> adjacencyMatrix(numberPixels+2,Vertex());

//Weight Assignment:


  

  out_image = gradient.clone ();

  //int adjacencyMatrix[height][width];

  for (int i = 0; i < width; i++)
    {

      for (int j = 0; j < height; j++)
    {


//int intensity = abs(out_image.at<uchar>(j,i));
	 // if (Scalar.val(intensity) < 100)

Vec3b pixel = out_image.at < Vec3b > (j, i);

      if (pixel[0] < 100 && pixel[1] < 100 && pixel[2] < 100)

//cout<<"intensity"<<intensity;
//if(intensity < 100)	   
 {
	      adjacencyMatrix[j][i] = INT_MAX;
	    }
	  else
	    {
	      adjacencyMatrix[j][i] = 0;
	    }
	}
}
cout<<"After Weight Part.";
      //Mat final_image;

      finalImg = Mat::zeros (out_image.rows, out_image.cols, CV_8UC3);

      int n;
      f >> n;

//=======================================================================================
for (int i = 0; i < height; ++i)
    {

      for (int j = 0; j < width; ++j){

int inten;
inten = 0;

//Scalar intensity1 = gray_image.at<uchar>(j, i);
//    cout << "Intensity = " << endl << " " << intensity1.val[0] << endl << endl;

//if(inten > (int)intensity1.val[0]){
//if(inten >= max_intensity){
//outputFile << inten << endl;
//max_intensity = inten;
//}
}
}
//cout<<"Max Intensity: "<<max_intensity;

//outputFile.close();

//Ends here.
int countForward = 0;
int countBackground = 0;
int intensityForward = 0;
int intensityBackground = 0;

/*cout<<"intensityForward: "<<intensityForward<<endl;
cout<<"countForward: "<<countForward<<endl;
cout<<"intensityBackground: "<<intensityBackground<<endl;
cout<<"countBackground: "<<countBackground<<endl;
*/

//int matrix[4][4];
int matrix[4][4];
int delta[4][4];
int weightedMatrix[16][16];

matrix[0][0] = 1;
matrix[0][1] = 2;
matrix[0][2] = 20;
matrix[0][3] = 1;
matrix[1][0] = 1;
matrix[1][1] = 100;
matrix[1][2] = 5;
matrix[1][3] = 30;
matrix[2][0] = 1;
matrix[2][1] = 3;
matrix[2][2] = 40;
matrix[2][3] = 1;
matrix[3][0] = 1;
matrix[3][1] = 4;
matrix[3][2] = 2;
matrix[3][3] = 1;

int deltaTest[2][2];

int weightedMatrixTest[4][4];

deltaTest[0][0] = 1;
deltaTest[0][1] = 52;
deltaTest[1][0] = 74;
deltaTest[1][1] = 100;

int maxDifference =0;
int difference =0;

cout<<"0"<<endl;


for(int i=0;i<4;i++){
for(int j=0;j<4;j++){
difference =0;
for (int k = -1; k <= 1; k++) {
                    for (int l = -1; l <= 1; l++){

if((i+k)> -1 && (l+j) > -1 && (i+k) < 4 && (l+j) < 4) {
                        difference = matrix[i][j] - matrix[k + i][l + j];

//cout<<endl<<abs(difference)<<endl;
if(abs(difference) > maxDifference){
maxDifference = abs(difference);
}
}
}
}

delta[i][j] = abs(difference);

}
}

cout<<"Diff Matrix Test."<<endl;

for(int i=0;i<2;i++){
for(int j=0;j<2;j++){

//cout<<deltaTest[i][j]<<" ";
}
cout<<endl;
}


//cout<<"0.5 "<<endl;


cout<<"   1";

//Weighted Matrix Test
//max_intensity = 100;
//max_intensity = maxDifference;
//=============================
/*for(int i=0;i<numberPixels;++i){
        int xValue = i/width;
        int yValue = i%width;
        if(i-1>=0 && yValue-1>=0){//edge to left
            double edge_weight = abs(gradient_image.at<uchar>(xValue,yValue) - gradient_image.at<uchar>(xValue,yValue-1));
            if(edge_weight<0.5){
                adjacencyMatrix.at(i).addEdge(Edge(i,i-1,9999,false,false));
            }else{
                adjacencyMatrix.at(i).addEdge(Edge(i,i-1,1.0,false,false));
            }
        }
        if(i+1<numberPixels && yValue+1<width){//edge to right
            double edge_weight = abs(gradient_image.at<uchar>(xValue,yValue) - gradient_image.at<uchar>(xValue,yValue+1));
            if(edge_weight<0.5){
                adjacencyMatrix.at(i).addEdge(Edge(i,i+1,9999,false,false));
            }else{
                adjacencyMatrix.at(i).addEdge(Edge(i,i+1,1.0,false,false));
            }
        }
        if(i-width>=0 && xValue-1>=0){//edge to top
            double edge_weight = abs(gradient_image.at<uchar>(xValue,yValue) - gradient_image.at<uchar>(xValue-1,yValue));
            if(edge_weight<0.5){
                adjacencyMatrix.at(i).addEdge(Edge(i,i-width,9999,false,false));
            }else{
                adjacencyMatrix.at(i).addEdge(Edge(i,i-width,1.0,false,false));
            }
        }
        if(i+width<numberPixels && xValue+1<height){//edge to bottom
            double edge_weight = abs(gradient_image.at<uchar>(xValue,yValue) - gradient_image.at<uchar>(xValue+1,yValue));
            if(edge_weight<0.5){
                adjacencyMatrix.at(i).addEdge(Edge(i,i+width,9999,false,false));
            }else{
                adjacencyMatrix.at(i).addEdge(Edge(i,i+width,1.0,false,false));
            }
        }
    }
*/
//=======================================================================================

cout<<"Pixel Parts";
int count = INT_MAX;
int x,y,t;
      // get the initil pixels
      for (int i = 0; i < n; ++i)
	{
	  f >> x >> y >> t;

      if (x < 0 || x >= width || y < 0 || y >= height)
        {
          cout << "I valid pixel mask!" << endl;
          return -1;
        }
     // int indexValue = y * width + x % width;

	  if (t == 1)
	    {
//int count = 21474834;

while(count > 1){
flood(x, y, height, width);
count--;
}
}
	}
      imwrite (argv[3], out_image);
//        out_image.at<Vec3b>(y, x) = pixel;

    //  imshow ("Before Calculation: out Image: ", out_image);
     // imshow ("Before Calculation: In Image: ", in_image);
     // imshow ("Before Calculation: Gradient Image: ", gradient_image);

      // write it on disk
      imwrite (argv[3], finalImg);

      // also display them both

      namedWindow ("Original image", WINDOW_AUTOSIZE);
      namedWindow ("Show Marked Pixels", WINDOW_AUTOSIZE);
      imshow ("Original image", in_image);
      imshow ("Show Marked Pixels", finalImg);   
//	imshow ("Show Marked P22ixels 2", final_image);
      waitKey (0);
      return 0;
    }



