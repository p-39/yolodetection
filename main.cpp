#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

int inpWidth = 416;
int inpHeight = 416;

string getFileName();
vector<string> loadClassNames(const string& classNameFile);
void drawBox(Mat& frame, int classId, float conf, int tlx, int tly, int brx, int bry);
void processDetections(const vector<Mat>& outs, Mat& frame);


int main(int argc, const char** argv) {
	string classFile = "C:/Users/parth kadia/source/repos/Project1/Project1/assets/coco.names.txt";
	string modelConfiguration = "C:/Users/parth kadia/source/repos/Project1/Project1/assets/yolov3.cfg";
	string modelWeights = "C:/Users/parth kadia/source/repos/Project1/Project1/assets/yolov3.weights";

	vector<string> classNames = loadClassNames(classFile);

	Net net = readNetFromDarknet(modelConfiguration, modelWeights);
	net.setPreferableBackend(DNN_BACKEND_OPENCV);
	net.setPreferableTarget(DNN_TARGET_CPU);

	string videoFile = getFileName();
	VideoCapture cap(videoFile);
	if (!cap.isOpened()) {
		cerr << "Problem while opening file" << endl;
		return -1;
	}
	Mat frame, blob;
	while (cap.read(frame)) {
		blobFromImage(frame, blob, 1 / 255.0, Size(inpWidth, inpHeight), Scalar(0, 0, 0), true, false);
		net.setInput(blob);
		vector<Mat> outs;
		net.forward(outs, net.getUnconnectedOutLayersNames());

		processDetections(outs, frame);

		imshow("Object-detection", frame);
		if (waitKey(10) == 27) break;
	}
	return 0;
}

string getFileName() {
	cout << "Enter the videofile path for which you want to DetectNTrack /n Press Enter to proceed with default" << endl;
	string videoFile;
	getline(cin, videoFile);
	if (videoFile.empty()) {
		return "C:/Users/parth kadia/source/repos/Project1/Project1/assets/testVideo.mp4";
	}
	return videoFile;
}

// Helper Function to load class names(labels)
vector<string> loadClassNames(const string& classNameFile) {
	ifstream inputfile(classNameFile.c_str());
	if (!inputfile.is_open()) {
		cerr << "Error: Unable to open the class names file: " << classNameFile << endl;
		exit(EXIT_FAILURE);
	}

	vector<string> classNames;
	string line;
	while (getline(inputfile, line)) {
		classNames.push_back(line);
	}
	return classNames;
}

// Helper Function to draw bounding boxes on detected objects
void drawBox(Mat& frame, int classId, float conf, int tlx, int tly, int brx, int bry) {
	rectangle(frame, Point(tlx, tly), Point(brx, bry), Scalar(255, 255, 255), 1);
	string conf_label = format("%.2f", conf);
	string label{};
	if (!label.empty()) {
		label = conf_label;
	}

	int baseline;
	Size labelSize = getTextSize(label, FONT_HERSHEY_PLAIN, 0.5, 1, &baseline);
	tly = max(tly, labelSize.height);
	rectangle(frame, Point(tlx, tly - labelSize.height), Point(tlx + labelSize.width, tly + baseline), Scalar(255, 255, 255), 1);
	putText(frame, label, Point(tlx, tly), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 1, LINE_AA);
}

void processDetections(const vector<Mat>& outs, Mat& frame) {
	for (size_t i = 0; i < outs.size(); ++i) {
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols) {
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > 0.5) { // Confidence threshold
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;
				drawBox(frame, classIdPoint.x, confidence, left, top, left + width, top + height);
			}
		}
	}
}
