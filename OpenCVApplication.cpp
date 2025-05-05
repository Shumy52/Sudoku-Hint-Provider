// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <math.h>
#include <algorithm>
using namespace std;

wchar_t* projectPath;

// From lab 8
Mat autoThreshold(const Mat& src) {
	int height = src.rows;
	int width = src.cols;
	int totalPixels = height * width;
	int* outputHistogram = (int*)calloc(256, sizeof(int));
	if (!outputHistogram) {
		printf("Memory allocation failed for histogram.\n");
		return Mat(); // Return an empty Mat in case of failure  
	}

	int mini = 255, maxi = 0;

	// Compute histogram and find min/max intensity  
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			uchar pixelVal = src.at<uchar>(i, j);
			outputHistogram[pixelVal]++;
			if (pixelVal < mini) mini = pixelVal;
			if (pixelVal > maxi) maxi = pixelVal;
		}
	}

	int T = (mini + maxi) / 2, newT = T;
	double allowedError = 0.1;

	do {
		T = newT;

		int nrSmol = 0, nrLarge = 0, sumSmol = 0, sumLarge = 0;
		for (int i = 0; i < 256; i++) {
			if (i < T) {
				nrSmol += outputHistogram[i];
				sumSmol += i * outputHistogram[i];
			}
			else {
				nrLarge += outputHistogram[i];
				sumLarge += i * outputHistogram[i];
			}
		}

		int meanSmol = nrSmol > 0 ? sumSmol / nrSmol : 0;
		int meanLarge = nrLarge > 0 ? sumLarge / nrLarge : 0;

		newT = (meanSmol + meanLarge) / 2;
	} while (abs(newT - T) > allowedError);

	Mat dst = Mat::zeros(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			dst.at<uchar>(i, j) = src.at<uchar>(i, j) > newT ? 255 : 0;
		}
	}

	free(outputHistogram);

	return dst;
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// HSV components
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// Defining pointers to each matrix (8 bits/pixels) of the individual components H, S, V 
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// Defining the pointer to the HSV image matrix (24 bits/pixel)
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;	// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}



/* Histogram display function - display a histogram using bars (simlilar to L3 / Image Processing)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
//void showHistogram(const string& name, int* hist, const int  hist_cols, const int hist_height)
//{
//	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image
//
//	//computes histogram maximum
//	int max_hist = 0;
//	for (int i = 0; i<hist_cols; i++)
//	if (hist[i] > max_hist)
//		max_hist = hist[i];
//	double scale = 1.0;
//	scale = (double)hist_height / max_hist;
//	int baseline = hist_height - 1;
//
//	for (int x = 0; x < hist_cols; x++) {
//		Point p1 = Point(x, baseline);
//		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
//		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
//	}
//
//	imshow(name, imgHist);
//}

// Code for projection computing

// Modify the showHistogram function to accept a vector<int> instead of int*  
void showHistogram(const string& name, const vector<int>& hist, const int hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image  

	// computes histogram maximum  
	int max_hist = *max_element(hist.begin(), hist.end());
	double scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta  
	}

	imshow(name, imgHist);
}



struct ProjectionProfile {
	vector<int> horiz;
	vector<int> vert;
};

ProjectionProfile computeProjectionProfile(const Mat& img) {

	int rows = img.rows;
	int cols = img.cols;

	ProjectionProfile profile;
	profile.horiz.resize(rows, 0);
	profile.vert.resize(cols, 0);

	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			if (img.at<uchar>(y, x) == 0) { // black pixel
				profile.horiz[y]++;
				profile.vert[x]++;
			}


    
	return profile;
}

string getProjectPath() {
	char buffer[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, buffer);
	return string(buffer);
}

// At runtime, we'll compute the projections for the "default" numbers, the ones we'll compare against.
map<int, ProjectionProfile> buildReferenceSet(const string& folderPath) {
	map<int, ProjectionProfile> refProfiles;

	for (int digit = 0; digit <= 9; ++digit) {
		
		// Example: C:\Projects\OpenCVApp\digits\0.png
		// So in the project files!!!
		
		string absolutePath = getProjectPath() + "\\" + folderPath;
		string filename = absolutePath + "\\" + to_string(digit) + ".png";

		printf("Filename where digits are searched: %s \n", filename.c_str());
		
		Mat img = imread(filename, IMREAD_GRAYSCALE);
		if (img.empty()) {
			cerr << "Failed to load digit " << digit << "\n";
			continue;
		}

		Mat resized;
		printf("Size of reference image is: rows %d, cols %d\n", img.rows, img.cols);
		cv::resize(img, resized, Size(28, 28), 0, 0, cv::INTER_NEAREST); // standard size for fairness
		// This is kind of content aware resizing. It's not just a crop.

		// Copy-pasted this because the accuracy is shit, and this made it better
		// I'll replace this with stuff we did at the lab. 
		resized = autoThreshold(resized);

		refProfiles[digit] = computeProjectionProfile(resized);
	}

	return refProfiles;
}

// This function will tell how far off from the default numbers we are
int projectionDistance(const ProjectionProfile& a, const ProjectionProfile& b) {
	int diff = 0;

	int minH = min(a.horiz.size(), b.horiz.size());
	for (int i = 0; i < minH; ++i)
		diff += abs(a.horiz[i] - b.horiz[i]);

	int minV = min(a.vert.size(), b.vert.size());
	for (int i = 0; i < minV; ++i)
		diff += abs(a.vert[i] - b.vert[i]);

	return diff;
}

bool isCellEmpty(const cv::Mat& cell, int blackThreshold = 30) {
	int blackPixels = cv::countNonZero(255 - cell);
	return blackPixels < blackThreshold;
}

																	// Why does it work only with >& ?
int recognizeDigit(const Mat& cell, const map<int, ProjectionProfile>& references) {
	
	// Might move this later, but I wanted to simplify the usage.
	// Instead of checking then calling recognize, I wanted to... just recognize
	// You can't? Then it's empty. One function call later down the line, no ifs
	if (isCellEmpty(cell)) {
		return -1;
	}

	Mat processed;

	cv::resize(cell, processed, Size(28, 28), 0, 0, cv::INTER_NEAREST);  // must match reference size
	
	cv::threshold(processed, processed, 128, 255, THRESH_BINARY); // I'' use the system call for now.
	// Safer than my damn lab code anyways.
	// If as a lab teacher you're reading this function... remind me to change that to autoThreshold

	ProjectionProfile query = computeProjectionProfile(processed);

	int bestDigit = -1;
	int bestScore = INT_MAX;

	// Replace the problematic structured binding with a standard pair iteration
	for (const auto& reference : references) {
		int digit = reference.first; // Extract the key
		const ProjectionProfile& refProfile = reference.second; // Extract the value
		int dist = projectionDistance(query, refProfile);
		if (dist < bestScore) {
			bestScore = dist;
			bestDigit = digit;
		}
	}

	return bestDigit;
}



vector<int> findLineCenters(const vector<int>& projection, int threshold) {
	vector<int> centers;
	bool inLine = false;
	int start = 0;
	//printf("Projection size is: %d\n", projection.size());
	for (int i = 0; i < projection.size(); ++i) {
		if (projection[i] > threshold) {
			if (!inLine) {
				inLine = true;
				start = i;
			}
		}
		else {
			if (inLine) {
				inLine = false;
				int end = i;
				int center = (start + end) / 2;
				printf("Line found at: %d\n", center);
				centers.push_back(center);
			}
		}
	}

	// Sometimes it's not catching the last line, when the end line ends with the projection itself
	// So we never get "out" of the line, required to register one, see "else" above. 
	// So if we're still in the line at the end of the projection, we consider the last possible value
	// the end of the line. Poetic, ain't it.
	if (inLine) {
		int end = static_cast<int>(projection.size()) - 1;
		int center = (start + end) / 2;
		printf("Line found at end: %d\n", center);
		centers.push_back(center);
	}

	return centers;
}

vector<Mat> segmentSudokuGrid(const Mat& binaryImage) {

	int rows = binaryImage.rows;
	int cols = binaryImage.cols;

	// 1. Compute projections
	vector<int> horProj(rows, 0), verProj(cols, 0);

	for (int y = 0; y < rows; ++y)
		for (int x = 0; x < cols; ++x)
			if (binaryImage.at<uchar>(y, x) == 0) {
				horProj[y]++;
				verProj[x]++;
			}

	// 2. Find line positions
	// Meaning of value: how long should the line be? At least X% of the length/width of the image
	int horThresh = 0.4 * cols;  
	int verThresh = 0.4 * rows;  

	vector<int> yLines = findLineCenters(horProj, horThresh);
	vector<int> xLines = findLineCenters(verProj, verThresh);

	// 3. Check we got 10 lines (i.e., 9 cells between them)
	if (xLines.size() != 10 || yLines.size() != 10) {
		cerr << "ERROR: Failed to detect 10 grid lines in each direction.\n";
		cerr << "Got " << xLines.size() << " vertical and " << yLines.size() << " horizontal lines.\n";
		return {};
	}

	// 4. Cut into 81 cells
	vector<Mat> cells;
	for (int row = 0; row < 9; ++row) {
		for (int col = 0; col < 9; ++col) {
			int x1 = xLines[col];
			int x2 = xLines[col + 1];
			int y1 = yLines[row];
			int y2 = yLines[row + 1];

			Rect cellRect(x1, y1, x2 - x1, y2 - y1);
			Mat cell = binaryImage(cellRect).clone();  // clone so we can safely use it later

			// Clip the borders to remove black edges
			int borderSize = static_cast<int>(0.1 * min(cell.rows, cell.cols)); // Adjust as needed
			Rect clippedRect(borderSize, borderSize, cell.cols - 2 * borderSize, cell.rows - 2 * borderSize);
			if (clippedRect.width > 0 && clippedRect.height > 0) {
				cell = cell(clippedRect).clone();
			}

			cells.push_back(cell);
		}
	}

	return cells;
}


void sudoku() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		printf("Size of input image: %d rows, %d cols", src.rows, src.cols);
		Mat srcPreprocessed = autoThreshold(src); // Threshold the image


		vector<Mat> cells = segmentSudokuGrid(srcPreprocessed);

		if (cells.size() == 81) {
			for (int i = 0; i < 81; ++i) {
				string filename = "cell_" + to_string(i) + ".png";
				imwrite(filename, cells[i]);  // Save to check if they look good
			}
		}
		else {
			cerr << "Error in sudoku function: number of cells is not 81 (9x9)";
			cerr << "Actual number: " << cells.size();
		}

		auto references = buildReferenceSet("Digits"); // From the "digits" directory in the PROJECT FOLDER

		for (size_t i = 0; i < cells.size(); ++i) {
			int recognized = recognizeDigit(cells[i], references);
			if (recognized == -1) {
				cout << "Cell " << i << "is empty" << "\n";
			}
			else {
				cout << "Cell " << i << ": Recognized digit: " << recognized << "\n";

			}
		}

		imshow("Thresholded input", srcPreprocessed);
		waitKey();
	}
}

int main() 
{
	utils::logging::setLogLevel(utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);

	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Primary function\n ");
		printf(" 2 - DEBUG: function X (todo)");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				sudoku();
				break;

			default:
				printf("Dumbass");
		}
	}
	while (op!=0);
	return 0;
}