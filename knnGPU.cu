#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "cublas_v2.h"
#include <chrono>
#include <random>
#include </root/cuda-workspace/knnGPU/src/deviceheader.h>

using namespace std;
using namespace cv;
// C(m,n) = A(m,k) * B(k,n)

cudaStream_t stream[900];

//Variables for each map/featuremaps
struct Map {
	float **map;
	float *d_map;

	float **weights;
	float *d_weights;

	float **target;
	float *d_target;

	float **delta;
	float *d_delta;

	float **deltabuffer;
	float *d_deltabuffer;

	float **deltazeropadded;
	float *d_deltazeropadded;

};

int richtig;
int falsch;

class Layer {
public:
	int type;
	int amount;
	int height;
	int width;
	int wheight; //WeightsHeight
	int wwidth;  //WeightsWidth

	float error = 0.0;

	vector<Map> Maps;

	Layer(int t, int a, int h, int w, int wh, int ww, float var) {
		unsigned seed = time(NULL);
		std::default_random_engine generator(seed);
		type = t;
		amount = a;
		height = h;
		width = w;
		wheight = wh;
		wwidth = ww;

		//Create as many internal Layers as needed
		for (int c = 0; c < amount; c++) {

			Maps.push_back(Map());

			//Create Map
			Maps[c].map = new float*[height];
			for (int i = 0; i < height; i++) {
				Maps[c].map[i] = new float[width];
			}

			//Create Weights
			Maps[c].weights = new float*[wheight];
			for (int i = 0; i < wheight; i++) {
				Maps[c].weights[i] = new float[wwidth];
			}

			//Initial Weights
			for (int i = 0; i < wheight; i++) {
				for (int e = 0; e < wwidth; e++) {

					std::normal_distribution<float> distribution(0.0,
							sqrt(var));

					Maps[c].weights[i][e] = distribution(generator);
				}
			}

			//Create Target
			if (type == 2) {
				Maps[c].target = new float*[height];
				for (int i = 0; i < height; i++) {
					Maps[c].target[i] = new float[width];
				}

				//Allocate Space for Target
				cudaMalloc((void**) &Maps[c].d_target, height * width * 4);
			}

			//Create delta
			Maps[c].delta = new float*[h];
			for (int i = 0; i < h; i++) {
				Maps[c].delta[i] = new float[w];
			}

			Maps[c].deltabuffer = new float*[wheight];
			for (int i = 0; i < wheight; i++) {
				Maps[c].deltabuffer[i] = new float[wwidth];
			}

			if (type == 3) {
				Maps[c].deltazeropadded = new float*[height + 4];
				for (int i = 0; i < height + 4; i++) {
					Maps[c].deltazeropadded[i] = new float[width + 4];
				}

				cudaMalloc((void**) &Maps[c].d_deltazeropadded,
						(height + 4) * (width + 4) * 4);
			}

			cudaMalloc((void**) &Maps[c].d_deltabuffer, wheight * wwidth * 4);

			cudaMalloc((void**) &Maps[c].d_delta, height * width * 4);

			//Allocate Space for Map on GPU
			cudaMalloc((void**) &Maps[c].d_map, height * width * 4);

			//Copy Weights to GPU
			cudaMalloc((void**) &Maps[c].d_weights, wheight * wwidth * 4);
			for (int i = 0; i < wheight; i++) {
				cudaMemcpy(Maps[c].d_weights + i * wwidth, Maps[c].weights[i],
						wwidth * 4, cudaMemcpyHostToDevice);
			}

		}

		cout << "Type: " << type << endl;
		cout << "Maps: " << amount << endl;
		cout << "Height x Width: " << height << "x" << width << endl;
		cout << "Weightsheight x Weightswidth: " << wheight << "x" << wwidth
				<< endl;
		cout << "Initialized" << endl << endl << endl;

	}

	virtual void loadData(int a, int b, Layer*& network) = 0;
	virtual void showLayer() = 0;

	void feedForward(Layer*& network) {

		if (network->type == 1 || network->type == 2) {
			for (int t = 0; t < network->amount; t++) {
				resetmap<<<network->height, network->width>>>(
						network->Maps[t].d_map);
			}
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					fcForward<<<wheight, wwidth, 0,
							stream[network->amount * t + c]>>>(Maps[c].d_map,
							Maps[c].d_weights, network->Maps[t].d_map, wheight,
							wwidth);
				}
			}

			for (int t = 0; t < network->amount; t++) {
				sigmoid<<<network->height, network->width>>>(
						network->Maps[t].d_map);
			}

		}
		if (network->type == 3) {
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					cvForward<<<network->height, network->width, 0,
							stream[network->amount * c + t]>>>(Maps[c].d_map,
							Maps[c].d_weights, network->Maps[t].d_map, width,
							wwidth);
				}
			}
			for (int t = 0; t < network->amount; t++) {
				sigmoid<<<network->height, network->width>>>(
						network->Maps[t].d_map);
			}
		}

		if (network->type == 4) {
			for (int c = 0; c < amount; c++) {
				maxForward<<<network->height, network->width>>>(Maps[c].d_map,
						network->Maps[c].d_map, width);
			}
		}

		if (network->type == 2) {
			for (int t = 0; t < network->amount; t++) {
				outputdeltaGPU<<<network->height, network->width>>>(
						network->Maps[t].d_delta, network->Maps[t].d_map,
						network->Maps[t].d_target);
			}
		}

	}

	void backpropagation(Layer*& network) {

		if (type == 1 || type == 2) {
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					fcBackpropagation<<<network->height, network->width, 0,
							stream[network->amount * c + t]>>>(Maps[c].d_delta,
							network->Maps[t].d_weights,
							network->Maps[t].d_delta, network->Maps[t].d_map,
							network->width * network->height, width * height);
				}
			}
		}

		if (type == 3) {
			for (int c = 0; c < amount; c++) {

				zeropadding<<<height + 4, width + 4>>>(
						Maps[c].d_deltazeropadded, Maps[c].d_delta, width,
						network->wwidth);
			}
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					cvBackward<<<network->height, network->width, 0,
							stream[network->amount * c + t]>>>(
							Maps[c].d_deltazeropadded,
							network->Maps[t].d_weights,
							network->Maps[t].d_delta, network->Maps[t].d_map,
							(width + 4), network->wwidth);
				}
			}
		}
		if (type == 4) {
			for (int t = 0; t < network->amount; t++) {

				resetmap<<<network->height, network->width>>>(
						network->Maps[t].d_delta);
			}
			for (int t = 0; t < network->amount; t++) {

				maxBackward<<<height, width>>>(network->Maps[t].d_map,
						network->Maps[t].d_delta, Maps[t].d_delta,
						network->width);
			}
		}

	}

	void update(Layer*& network) {

		if (type == 1 || type == 2) {
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					fcUpdate<<<network->width * network->height, width * height>>>(
							network->Maps[t].d_weights, Maps[c].d_delta,
							network->Maps[t].d_map);
				}
			}
		}

		if (type == 3) {
			for (int t = 0; t < network->amount; t++) {
				for (int c = 0; c < amount; c++) {
					cvUpdate<<<height, width, 0, stream[network->amount * c + t]>>>(
							network->Maps[t].d_deltabuffer, Maps[c].d_delta,
							network->Maps[t].d_map, network->width,
							network->wwidth);
				}
			}
			for (int t = 0; t < network->amount; t++) {

				cvAdd<<<network->wheight, network->wwidth>>>(
						network->Maps[t].d_deltabuffer,
						network->Maps[t].d_weights);
			}
		}
	}

	virtual ~Layer() {
	}

};

class Input: public Layer {
public:

	Input(int t, int a, int h, int w, int wh, int ww, float var) :
			Layer(t, a, h, w, wh, ww, var) {

	}

	void loadData(int a, int b, Layer*& network) {
		Mat image = imread(
				"/root/cuda-workspace/knnGPU/src/dataset/" + to_string(a) + "/"
						+ to_string(b) + ".jpg", 0);
		for (int c = 0; c < amount; c++) {
			for (int x = 0; x < image.size().height; x++) {
				for (int y = 0; y < image.size().width; y++) {
					int val = static_cast<float>(image.at<uchar>(x, y));

					Maps[c].map[x][y] = val / 255;
				}
			}
			for (int i = 0; i < height; i++) {
				cudaMemcpy(Maps[c].d_map + i * width, Maps[c].map[i], width * 4,
						cudaMemcpyHostToDevice);
			}

			for (int i = 0; i < network->width; i++) {
				if (i == a) {
					network->Maps[c].target[0][i] = 1;
				} else {
					network->Maps[c].target[0][i] = 0;
				}
			}
			for (int i = 0; i < network->height; i++) {
				cudaMemcpy(network->Maps[c].d_target + i * network->width,
						network->Maps[c].target[i], network->width * 4,
						cudaMemcpyHostToDevice);
			}
		}

	}

	void showLayer() {
		for (int c = 0; c < amount; c++) {
			for (int i = 0; i < height; i++) {
				cudaMemcpy(Maps[c].map[i], Maps[c].d_map + i * width, width * 4,
						cudaMemcpyDeviceToHost);
			}

			unsigned char res[28][28];
			for (int i = 0; i < 28; i++) {
				for (int e = 0; e < 28; e++) {
					res[i][e] = Maps[c].map[i][e];
				}
			}

			Mat src = Mat(28, 28, CV_8UC1, res);
			imshow("Original", src);
			//waitKey(0);
		}

	}

};

class Fullyconnected: public Layer {
public:

	Fullyconnected(int t, int a, int h, int w, int wh, int ww, float var) :
			Layer(t, a, h, w, wh, ww, var) {
	}

	void loadData(int a, int b, Layer*& network) {
		//NOT USED IN THIS LAYER
	}

	void showLayer() {
		/*for (int c = 0; c < amount; c++) {
		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].map[i], Maps[c].d_map + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].delta[i], Maps[c].d_delta + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < width; i++) {
		 //cout << Maps[c].map[0][i] << "   " << Maps[c].delta[0][i] << endl;
		 }
		 cout << endl;
		 }*/
	}

};

class Convolution: public Layer {
public:

	Convolution(int t, int a, int h, int w, int wh, int ww, float var) :
			Layer(t, a, h, w, wh, ww, var) {
	}

	void loadData(int a, int b, Layer*& network) {
		//NOT USED IN THIS LAYER
	}

	void showLayer() {

		/*for (int c = 0; c < amount; c++) {
		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].map[i], Maps[c].d_map + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].delta[i], Maps[c].d_delta + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < height + 4; i++) {
		 cudaMemcpy(Maps[c].deltazeropadded[i], Maps[c].d_deltazeropadded + i * (width + 4), (width + 4) * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < width; i++) {
		 for (int e = 0; e < height; e++) {
		 cout << Maps[c].map[e][i] << " ";

		 }
		 cout << endl;
		 }

		 cout << endl;

		 }*/
	}

};

class Maxpooling: public Layer {
public:

	Maxpooling(int t, int a, int h, int w, int wh, int ww, float var) :
			Layer(t, a, h, w, wh, ww, var) {
	}

	void loadData(int a, int b, Layer*& network) {
		//NOT USED IN THIS LAYER
	}

	void showLayer() {

		/*for (int c = 0; c < amount; c++) {
		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].map[i], Maps[c].d_map + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }

		 for (int i = 0; i < height; i++) {
		 cudaMemcpy(Maps[c].delta[i], Maps[c].d_delta + i * width, width * 4, cudaMemcpyDeviceToHost);
		 }



		 for (int i = 0; i < width; i++) {
		 for (int e = 0; e < height; e++) {
		 cout << Maps[c].map[e][i] << " ";

		 }
		 cout << endl;
		 }

		 cout << endl;

		 }*/
	}

};

class Output: public Layer {
public:

	Output(int t, int a, int h, int w, int wh, int ww, float var) :
			Layer(t, a, h, w, wh, ww, var) {
	}

	void loadData(int a, int b, Layer*& network) {
		//NOT USED IN THIS LAYER
	}

	void showLayer() {
		for (int c = 0; c < amount; c++) {
			error = 0.0;

			for (int i = 0; i < height; i++) {
				cudaMemcpy(Maps[c].map[i], Maps[c].d_map + i * width, width * 4,
						cudaMemcpyDeviceToHost);
			}

			for (int i = 0; i < height; i++) {
				cudaMemcpy(Maps[c].target[i], Maps[c].d_target + i * width,
						width * 4, cudaMemcpyDeviceToHost);
			}

			for (int i = 0; i < height; i++) {
				cudaMemcpy(Maps[c].delta[i], Maps[c].d_delta + i * width,
						width * 4, cudaMemcpyDeviceToHost);
			}

			for (int i = 0; i < width; i++) {
				error += 0.5 * (Maps[c].target[0][i] - Maps[c].map[0][i])
						* (Maps[c].target[0][i] - Maps[c].map[0][i]);
			}

			for (int i = 0; i < width; i++) {
			 cout << Maps[c].map[0][i] << "  " << Maps[c].target[0][i]
			 << "   " << Maps[c].delta[0][i] << endl;
			 }
			int result = 0;
			for (int i = 1; i < 10; i++) {
				if (Maps[0].map[0][i]
						>Maps[0].map[0][i - 1]
						&& Maps[0].map[0][i]
								> Maps[0].map[0][result]) {
					result = i;
				}

			}

			if(Maps[c].target[0][result] == 1){
				cout << "Richtig" << endl;
				richtig++;
			}else{
				cout << "Falsch" << endl;
				falsch++;
			}
			cout << endl;
			cout << error << endl << endl;

		}
	}

};

int imgHeight = 28;
int imgWidth = 28;

int samples = 300;
int outs = 10;

struct data {
	float **red;
	float *d_red;

	float **green;
	float *d_green;

	float **blue;
	float *d_blue;
};

vector<vector<data> > dataset(samples, vector<data>(outs));

void ldata(int i, int e) {
	dataset[i][e].red = new float*[imgHeight];
	for (int f = 0; f < imgHeight; f++) {
		dataset[i][e].red[f] = new float[imgWidth];
	}

	dataset[i][e].green = new float*[imgHeight];
	for (int f = 0; f < imgHeight; f++) {
		dataset[i][e].green[f] = new float[imgWidth];
	}
	dataset[i][e].blue = new float*[imgHeight];
	for (int f = 0; f < imgHeight; f++) {
		dataset[i][e].blue[f] = new float[imgWidth];
	}
	//////////////////////////////////////

	Mat image = imread(
			"/root/cuda-workspace/knnGPU/src/dataset/" + to_string(e) + "/"
					+ to_string(i) + ".jpg", 1);

	for (int x = 0; x < image.size().height; x++) {
		for (int y = 0; y < image.size().width; y++) {
			//load data into array and normalize values
			dataset[i][e].blue[x][y] = image.at<cv::Vec3b>(y, x)[0] / 255;
			dataset[i][e].green[x][y] = image.at<cv::Vec3b>(y, x)[1] / 255;
			dataset[i][e].red[x][y] = image.at<cv::Vec3b>(y, x)[2] / 255;

		}

	}
	//Allocate Device Memory on device (GPU)
	cudaMalloc((void**) &dataset[i][e].d_red, imgWidth * imgHeight * 4);
	cudaMalloc((void**) &dataset[i][e].d_green, imgWidth * imgHeight * 4);
	cudaMalloc((void**) &dataset[i][e].d_blue, imgWidth * imgHeight * 4);

}

void cdata(int i, int e) {
	for (int f = 0; f < imgHeight; f++) {
		//copy data into device (GPU)
		cudaMemcpy(dataset[i][e].d_red + f * imgWidth, dataset[i][e].red[f],
				imgWidth * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dataset[i][e].d_green + f * imgWidth, dataset[i][e].green[f],
				imgWidth * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(dataset[i][e].d_blue + f * imgWidth, dataset[i][e].blue[f],
				imgWidth * sizeof(float), cudaMemcpyHostToDevice);
	}
}

vector<Layer*> network;
int topology[5][6] = { { 0, 3, imgHeight, imgWidth, 5, 5 }, { 3, 5, 24, 24, 0,
		0 }, { 4, 5, 12, 12, 12 * 12, 100 }, { 1, 1, 1, 100, 100, outs }, { 2,
		1, 1, outs, 0, 0 } };


int length = (sizeof(topology) / sizeof(topology[0]));

//Interface
Mat img = Mat(1080, 1920, CV_8UC3, Scalar(0, 0, 0));
bool down = false;
Mat cropped;
Mat gray;
Mat digit = Mat(28, 28, CV_8UC3, Scalar(0, 0, 0));
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
	if (event == EVENT_LBUTTONDOWN) {
		down = true;
	}

	if (event == EVENT_LBUTTONUP) {
		down = false;
	}

	if (event == EVENT_MOUSEMOVE) {
		if (down == true) {
			circle(img, Point(x, y), 20, Scalar(255, 255, 255), CV_FILLED, 8,
					0);
		}
		imshow("Drawing Window", img);
	}

	if (event == EVENT_LBUTTONUP) {

		cvtColor(img, gray, CV_BGR2GRAY);
		//threshold(gray, gray, 0, 55, THRESH_BINARY_INV);
		int largest_area = 0;
		int largest_contour_index = 0;
		Rect bounding_rect;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		findContours(gray, contours, hierarchy, CV_RETR_CCOMP,
				CV_CHAIN_APPROX_SIMPLE);
		for (int i = 0; i < contours.size(); i++) {
			double a = contourArea(contours[i], false);
			if (a > largest_area) {
				largest_area = a;
				cout << i << " area  " << a << endl;
				largest_contour_index = i;
				bounding_rect = boundingRect(contours[i]);
			}
		}

		img(bounding_rect).copyTo(cropped); //Zahl wird ausgeschnitten und in "cropped" kopiert
		resize(cropped, digit, cvSize(28, 28));
		imshow("Digit", digit);
		imwrite("/root/cuda-workspace/knnGPU/src/dataset/999/0.jpg", digit);

		ldata(0, 999); 		//load cropped digit
		cdata(0, 999);		//copy cropped digit into device

		//set digit as Input (3 channels)
		setData<<<imgWidth, imgWidth>>>(dataset[0][999].d_red,
				network[0]->Maps[0].d_map);
		setData<<<imgWidth, imgWidth>>>(dataset[0][999].d_green,
				network[0]->Maps[1].d_map);
		setData<<<imgWidth, imgWidth>>>(dataset[0][999].d_green,
				network[0]->Maps[2].d_map);

		for (int i = 0; i < length - 1; i++) {
			network[i]->feedForward(network[i + 1]);
		}

		for (int i = 0; i < 1; i++) {
			cudaMemcpy(network[length - 1]->Maps[0].map[i],
					network[length - 1]->Maps[0].d_map + i * outs, outs * 4,
					cudaMemcpyDeviceToHost);
		}

		for (int i = 0; i < outs; i++) {
			cout << network[length - 1]->Maps[0].map[0][i] << endl;
		}
		cout << endl << endl;
		int result = 0;
		for (int i = 1; i < outs; i++) {
			if (network[length - 1]->Maps[0].map[0][i]
					> network[length - 1]->Maps[0].map[0][i - 1]
					&& network[length - 1]->Maps[0].map[0][i]
							> network[length - 1]->Maps[0].map[0][result]) {
				result = i;
			}

		}
		rectangle(img, Point(10, 10), Point(100, 40), Scalar(0, 0, 0), -1, 8);
		putText(img, "Zahl: " + to_string(result), cvPoint(30, 30),
				FONT_HERSHEY_COMPLEX_SMALL, 0.8, cvScalar(0, 0, 255), 1, CV_AA);

	}

	if (event == EVENT_RBUTTONDOWN) {
		img = Scalar(0, 0, 0);
	}

}

int main() {

	//Create 900 cudastreams
	for (int i = 0; i < 900; i++) {
		cudaStreamCreate(&stream[i]);
	}

	//load data from dataset into program
	for (int i = 0; i < samples; i++) {
		for (int e = 0; e < outs; e++) {
			ldata(i, e);
		}
	}

	//copy data into device
	for (int i = 0; i < samples; i++) {
		for (int e = 0; e < outs; e++) {
			cdata(i, e);

		}
	}

	//Initialize Network

	float var;
	for (int i = 0; i < length; i++) {

		//Calculate Variation for Xavier Initialization
		var = 0.0;

		if (topology[i + 1][0] == 3) {
			var = topology[i][4] * topology[i][5] * topology[i][1];
		}

		if (topology[i + 1][0] == 2 || topology[i + 1][0] == 1) {
			var = topology[i][4] * topology[i][1];
		}

		var = 1.0 / var;

		//Create Layerobject in the vector network with parameters from topology
		if (topology[i][0] == 0) {
			network.push_back(
					new Input(topology[i][0], topology[i][1], topology[i][2],
							topology[i][3], topology[i][4], topology[i][5],
							var));
		}

		if (topology[i][0] == 1) {
			network.push_back(
					new Fullyconnected(topology[i][0], topology[i][1],
							topology[i][2], topology[i][3], topology[i][4],
							topology[i][5], var));
		}
		if (topology[i][0] == 2) {
			network.push_back(
					new Output(topology[i][0], topology[i][1], topology[i][2],
							topology[i][3], topology[i][4], topology[i][5],
							var));
		}

		if (topology[i][0] == 3) {
			network.push_back(
					new Convolution(topology[i][0], topology[i][1],
							topology[i][2], topology[i][3], topology[i][4],
							topology[i][5], var));
		}

		if (topology[i][0] == 4) {
			network.push_back(
					new Maxpooling(topology[i][0], topology[i][1],
							topology[i][2], topology[i][3], topology[i][4],
							topology[i][5], var));
		}

	}
	for (int l = 0; l < 200; l++) {
		for (int b = 1; b < 190; b++) {
			for (int a = 0; a < outs; a++) {

				//Load Input and Target into the network
				setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_red,
						network[0]->Maps[0].d_map);
				setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_green,
						network[0]->Maps[1].d_map);
				setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_green,
						network[0]->Maps[2].d_map);

				setTarget<<<1, outs>>>(network[length - 1]->Maps[0].d_target,
						a);

				cudaDeviceSynchronize();
				//FeedForward
				for (int i = 0; i < length - 1; i++) {
					network[i]->feedForward(network[i + 1]);
				}

				//Backpropagate the error
				for (int i = length - 1; i > 0; i--) {
					network[i]->backpropagation(network[i - 1]);
				}

				//Update weights based on bp results
				for (int i = 1; i < length; i++) {
					network[i]->update(network[i - 1]);
				}
				//ShowMaps


			}
			cout << l << endl;

		}
	}

	for (int b = 190; b < 200; b++) {
		for (int a = 0; a < outs; a++) {
		setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_red,
				network[0]->Maps[0].d_map);
		setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_green,
				network[0]->Maps[1].d_map);
		setData<<<imgWidth, imgWidth>>>(dataset[b][a].d_green,
				network[0]->Maps[2].d_map);

		setTarget<<<1, outs>>>(network[length - 1]->Maps[0].d_target, a);

		for (int i = 0; i < length - 1; i++) {
						network[i]->feedForward(network[i + 1]);
					}
		for (int i = 0; i < length; i++) {
								network[i]->showLayer();
							}
		}


	}
	namedWindow("Drawing Window", 1);
	imshow("Drawing Window", img);
	setMouseCallback("Drawing Window", CallBackFunc, NULL);
	waitKey(0);
	cout << richtig << endl;
	cout << falsch << endl;

}

