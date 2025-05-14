// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <math.h>
#include <algorithm>
using namespace std;

bool DEBUG = false;

wchar_t* projectPath;

bool isInside(const Mat& img, int i, int j) {
	return (i >= 0 && i < img.rows && j >= 0 && j < img.cols);
}

Mat erosion(Mat src)
{
	int di[8] = { -1, -1, 0, 1, 1, 1, 0, -1 };
	int dj[8] = { 0, -1, -1, -1, 0, 1, 1, 1 };
	int height = src.rows;
	int width = src.cols;
	Mat dst = src.clone();

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			if (src.at<uchar>(i, j) == 255) { // If it's part of the background
				for (int k = 0; k < 8; k++) {
					int ni = i + di[k];
					int nj = j + dj[k];
					if (isInside(src, ni, nj) && src.at<uchar>(ni, nj) == 0) {
						dst.at<uchar>(ni, nj) = 255;
					}
				}
			}
		}
	}
	return dst;
}

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

// Bypass, not working momentarily
Mat skeletonize(const Mat& src) {
	Mat binaryImg;
	if (countNonZero(src == 0) > countNonZero(src == 255)) {
		binaryImg = src.clone();
	} else {
		binaryImg = 255 - src;
	}
	Mat thinned;
	thinned = binaryImg.clone();
	return thinned;
}


Mat centerDigit(const Mat& src, int outputSize = 28) {

	Mat inverted = 255 - src;

	vector<vector<Point>> contours;
	findContours(inverted, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	// If no contours found, return the original resized image
	if (contours.empty()) {
		Mat resized;
		resize(src, resized, Size(outputSize, outputSize));
		return resized;
	}

	// Find the largest contour (should be the digit)
	size_t largestContourIdx = 0;
	double largestArea = 0;
	for (size_t i = 0; i < contours.size(); i++) {
		double area = contourArea(contours[i]);
		if (area > largestArea) {
			largestArea = largestArea;
			largestContourIdx = i;
		}
	}

	// Get the bounding rectangle of the digit
	Rect boundingBox = boundingRect(contours[largestContourIdx]);

	// Create a new image with the digit
	Mat digitOnly = src(boundingBox);

	// Create output image (white background)
	Mat centered = Mat(outputSize, outputSize, CV_8UC1, Scalar(255));

	// Calculate position to place digit in center
	int offsetX = (outputSize - boundingBox.width) / 2;
	int offsetY = (outputSize - boundingBox.height) / 2;

	// Ensure offsets are non-negative
	offsetX = max(0, offsetX);
	offsetY = max(0, offsetY);

	// Create ROI in centered image
	int roiWidth = min(boundingBox.width, outputSize - offsetX);
	int roiHeight = min(boundingBox.height, outputSize - offsetY);
	Mat roi = centered(Rect(offsetX, offsetY, roiWidth, roiHeight));

	// Copy the digit to the centered position
	Mat resizedDigit;
	resize(digitOnly, resizedDigit, Size(roiWidth, roiHeight));
	resizedDigit.copyTo(roi);

	return centered;
}


Mat preprocessDigit(const Mat& cell, int size = 28, bool applySkeletonization = false) {

	Mat cleaned;
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
	morphologyEx(cell, cleaned, MORPH_OPEN, element);

	Mat centered = centerDigit(cleaned, size);

	// Apply skeletonization if requested (NOT WORKING)
	if (applySkeletonization) {
		Mat skeleton = skeletonize(centered);
		return skeleton;
	}

	return centered;
}


string getProjectPath() {
	char buffer[MAX_PATH];
	GetCurrentDirectoryA(MAX_PATH, buffer);
	return string(buffer);
}

// At runtime, we'll compute the projections for the "default" numbers, the ones we'll compare against.
map<int, ProjectionProfile> buildReferenceSet(const string& folderPath) {
	map<int, ProjectionProfile> refProfiles;

	for (int digit = 1; digit <= 9; ++digit) {
	
		// Example: C:\Projects\OpenCVApp\digits\0.png
		// So in the project files!!!
	
		string absolutePath = getProjectPath() + "\\" + folderPath;
		string filename = absolutePath + "\\" + to_string(digit) + ".png";

		if (DEBUG) {
			printf("Filename where digits are searched: %s \n", filename.c_str());

		}
	
		Mat img = imread(filename, IMREAD_GRAYSCALE);
		if (img.empty()) {
			cerr << "Failed to load digit " << digit << "\n";
			continue;
		}

		Mat preprocessed = preprocessDigit(img); // Preprocess the digit
	
		// Save the resized template to disk with a slightly changed name
		string resizedFilename = absolutePath + "\\" + to_string(digit) + "eroded.png";
		imwrite(resizedFilename, preprocessed);

		refProfiles[digit] = computeProjectionProfile(preprocessed);
	}

	return refProfiles;
}

// This function will tell how far off from the default numbers we are
int projectionDistance(const ProjectionProfile& a, const ProjectionProfile& b) {  
	int diffX = 0;  
	int diffY = 0;  

	int minH = min(a.horiz.size(), b.horiz.size());  
	for (int i = 0; i < minH; ++i)  
		diffX += 2 * abs(a.horiz[i] - b.horiz[i]); // Increase weight for horizontal differences  

	int minV = min(a.vert.size(), b.vert.size());  
	for (int i = 0; i < minV; ++i)  
		diffY += abs(a.vert[i] - b.vert[i]);  

	return diffX + diffY; 
}

bool isCellEmpty(const cv::Mat& cell, int blackThreshold = 20) {
	int blackPixels = cv::countNonZero(255 - cell);
	return blackPixels < blackThreshold;
}

int recognizeDigit(const Mat& cell, const map<int, ProjectionProfile>& references) {
	// Check if the cell is empty  
	if (isCellEmpty(cell)) {
		return -1;
	}

	Mat preprocessed = preprocessDigit(cell); // Preprocess the digit
	ProjectionProfile query = computeProjectionProfile(preprocessed);

	if (DEBUG) {
		cout << "Query Projection Profile (Horizontal): ";
		for (const auto& val : query.horiz) {
			cout << val << " ";
		}
		cout << "\nQuery Projection Profile (Vertical): ";
		for (const auto& val : query.vert) {
			cout << val << " ";
		}
		cout << "\n";
	}

	int bestDigit = -1;
	int bestScore = INT_MAX;

	for (const auto& reference : references) {
		int digit = reference.first;
		const ProjectionProfile& refProfile = reference.second;
		int dist = projectionDistance(query, refProfile);

		if (DEBUG) {
			cout << "Reference Digit: " << digit << "\n";
			cout << "Reference Projection Profile (Horizontal): ";
			for (const auto& val : refProfile.horiz) {
				cout << val << " ";
			}
			cout << "\nReference Projection Profile (Vertical): ";
			for (const auto& val : refProfile.vert) {
				cout << val << " ";
			}
			cout << "\n";
			cout << "Distance to Reference Digit " << digit << ": " << dist << "\n";

		}

		if (dist < bestScore) {
			bestScore = dist;
			bestDigit = digit;
		}
	}

	const int ABSOLUTE_THRESHOLD = 200; // Wiggle it around 150-300 

	if (bestScore > ABSOLUTE_THRESHOLD) {
		bestDigit = -1; // Not confident enough...
	}

	// Debugging: Print the best match  
	if (DEBUG) {
		cout << "Best Match: Digit " << bestDigit << " with score " << bestScore << "\n";
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
				if (DEBUG) {
					printf("Line found at: %d\n", center);
				}
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
		if (DEBUG) {
			printf("Line found at: %d\n", center);
		}
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

			cell = preprocessDigit(cell); // Preprocess the cell for digit recognition

			cells.push_back(cell);
		}
	}

	return cells;
}

bool isValid(int grid[9][9], int row, int col, int num) {
	for (int x = 0; x < 9; ++x)
		if (grid[row][x] == num || grid[x][col] == num)
			return false;
	int startRow = row - row % 3, startCol = col - col % 3;
	for (int i = 0; i < 3; ++i)
		for (int j = 0; j < 3; ++j)
			if (grid[startRow + i][startCol + j] == num)
				return false;
	return true;
}

void showSudokuWithHint(int grid[9][9], int hintRow, int hintCol, int hintNum) {
	int cellSize = 50;
	int imgSize = cellSize * 9;
	Mat img(imgSize, imgSize, CV_8UC3, Scalar(255, 255, 255));

	// Grid lines
	for (int i = 0; i <= 9; ++i) {
		int thickness = (i % 3 == 0) ? 2 : 1;
		line(img, Point(0, i * cellSize), Point(imgSize, i * cellSize), Scalar(0, 0, 0), thickness);
		line(img, Point(i * cellSize, 0), Point(i * cellSize, imgSize), Scalar(0, 0, 0), thickness);
	}

	// Digits
	for (int row = 0; row < 9; ++row) {
		for (int col = 0; col < 9; ++col) {
			int val = grid[row][col];
			if (val > 0) {
				putText(img, to_string(val), Point(col * cellSize + 15, row * cellSize + 35),
					FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 0), 2);
			}
		}
	}

	// Draw the hint in green
	if (hintRow != -1 && hintCol != -1 && hintNum != -1) {
		putText(img, to_string(hintNum), Point(hintCol * cellSize + 15, hintRow * cellSize + 35),
			FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 200, 0), 2);
	}
	else {
		cout << "No hint available.\n";
		if (DEBUG) {
			cout << "Hint Row: " << hintRow << ", Hint Col: " << hintCol << ", Hint Num: " << hintNum << "\n";
		}
		return;
		
	}

	imshow("Sudoku with Hint", img);
	//waitKey();
}

Mat rotateImage(const Mat& src, int k) {
	Mat dst;
	if (k == 1)      rotate(src, dst, ROTATE_90_CLOCKWISE);
	else if (k == 2) rotate(src, dst, ROTATE_180);
	else if (k == 3) rotate(src, dst, ROTATE_90_COUNTERCLOCKWISE);
	else             dst = src.clone();
	return dst;
}

void sudoku() {
	char fname[MAX_PATH];
	while (openFileDlg(fname)) {
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat srcPreprocessed = autoThreshold(src); 

		int bestRotation = 0, maxRecognized = -1;
		vector<Mat> bestCells;
		auto references = buildReferenceSet("Digits");

		for (int rot = 0; rot < 4; ++rot) {
			Mat rotated = rotateImage(src, rot);
			Mat srcPreprocessed = autoThreshold(rotated);
			vector<Mat> cells = segmentSudokuGrid(srcPreprocessed);

			if (cells.size() != 81) continue;

			int recognizedCount = 0;
			for (int i = 0; i < 81; ++i) {
				int recognized = recognizeDigit(cells[i], references);
				if (recognized != -1) recognizedCount++;
			}
			if (recognizedCount > maxRecognized) {
				maxRecognized = recognizedCount;
				bestRotation = rot;
				bestCells = cells;
			}
		}

		if (bestCells.size() == 81) {
			if (DEBUG) {
				for (int i = 0; i < 81; ++i) {
					string filename = "bestCells/cell_" + to_string(i) + ".png";
					imwrite(filename, bestCells[i]);  // Save to check if they look good
				}
			}
		}
		else {
			cerr << "Error in sudoku function: number of bestCells is not 81 (9x9)";
			cerr << "Actual number: " << bestCells.size();
		}

		if (DEBUG) {
			for (size_t i = 0; i < bestCells.size(); ++i) {
				int recognized = recognizeDigit(bestCells[i], references);
				if (recognized == -1) {
					cout << "Cell " << i << "is empty" << "\n";
				}
				else {
					cout << "Cell " << i << ": Recognized digit: " << recognized << "\n";

				}
			}
		}

		int sudokuGrid[9][9];
		for (int i = 0; i < 81; ++i) {
			sudokuGrid[i / 9][i % 9] = recognizeDigit(bestCells[i], references);
		}

		int hintRow = -1, hintCol = -1, hintNum = -1;
		for (int row = 0; row < 9 && hintNum == -1; ++row) {
			for (int col = 0; col < 9 && hintNum == -1; ++col) {
				if (sudokuGrid[row][col] == -1) { // empty cell
					for (int num = 1; num <= 9; ++num) {
						if (isValid(sudokuGrid, row, col, num)) {
							hintRow = row;
							hintCol = col;
							hintNum = num;
							break;
						}
					}
				}
			}
		}

		showSudokuWithHint(sudokuGrid, hintRow, hintCol, hintNum);

		imshow("Input", srcPreprocessed);
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
		printf("Welcome to McDonalds what can I get for ya:\n");
		printf(" 1 - Primary function\n");
		printf(" 2 - DEBUG MODE\n");
		printf(" 0 - I`ll have two number 9s, a number 9 large, a number 6 with extra dip, a num...\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				sudoku();
				break;
			case 2:
				DEBUG = true;
				sudoku();

			default:
				printf("Fuck off then");
		}
	}
	while (op!=0);
	return 0;
    
}