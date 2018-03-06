/*
 * test.cpp
 *
 *  Created on: May 16, 2016
 *      Author: lyan
 */

#define BOOST_FILESYSTEM_NO_DEPRECATED

#include <png++/png.hpp>
#include "image_statistic.h"
#include <cstdlib>

#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/progress.hpp"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include "boost/date_time/posix_time/posix_time.hpp"

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

struct stat st = {0};

namespace fs = boost::filesystem;

typedef boost::posix_time::ptime Time;
typedef boost::posix_time::time_duration TimeDuration;

using namespace std;
using namespace png;


//string path =
//		"/media/lyan/5C88875B4189CFED/KylbergTextureDataset-1.0-png-originals.7z.001.1/KylbergTextureDataset-1.0-png-originals/";
//string path = "/media/lyan/5C88875B4189CFED/noisy/KylbergTextureDataset-1.0-png-originals/";

string path = "/media/lyan/5C88875B4189CFED/noisy/";

string output_path = "noisy/";

extern "C++" double* calc_adjacency_matrix(int dm1, int dm2, int* c_values,
		int cols, int rows, int max_i, int max_j);
extern "C++" double* calc_signs(double* adj_matr, int cols, int rows, int dm1,
		int dm2, int max_i, int max_j);

extern "C++" double* calc_symmetric_adjacency_matrix(int* pic, int cols,
		int rows, int dm1, int dm2, int max_i, int max_j);

extern "C++" void test_matrix();

vector<string>* get_files_from_dir_with_noise(string noise_path) {
	boost::progress_timer t(std::clog);

	fs::path full_path(fs::initial_path<fs::path>());

	full_path = fs::system_complete(fs::path(path + noise_path));

	unsigned long file_count = 0;
	unsigned long dir_count = 0;
	unsigned long other_count = 0;
	unsigned long err_count = 0;

	vector<string> *files = new vector<string>;

	if (!fs::exists(full_path)) {
		std::cout << "\nNot found: " << full_path.filename() << std::endl;
		return files;
	}

	if (fs::is_directory(full_path)) {
		std::cout << "\nIn directory: " << full_path.filename() << "\n\n";
		fs::directory_iterator end_iter;
		for (fs::directory_iterator dir_itr(full_path); dir_itr != end_iter;
				++dir_itr) {
			try {
				if (fs::is_regular_file(dir_itr->status())) {
					++file_count;
					std::cout << dir_itr->path().filename() << "\n";
					files->push_back(dir_itr->path().filename().string());
				}

			} catch (const std::exception & ex) {
				++err_count;
				std::cout << dir_itr->path().filename() << " " << ex.what()
						<< std::endl;
			}
		}
		std::cout << "\n" << file_count << " files\n" << dir_count
				<< " directories\n" << other_count << " others\n" << err_count
				<< " errors\n";
	} else // must be a file
	{
		std::cout << "\nFound: " << full_path.filename() << "\n";
	}
	return files;
}

/**
 * here noise path stands for dispersion of noise in picture
 * assumed that pictures with noise of that dispersion are distributed among directories
 * respectively
 * path is path to target image
 * max i and j are maximum values for intensity
 * dm1 and dm2 are parameters for adjacency matrix
 */
double* exec(string noise_path, string path, string img_name, int max_i, int max_j, int dm1,
		int dm2) {
	cout << "exec started" << endl;
	cout << "current path:" << path << endl;
	cout << max_i << " " << max_j << " " << dm1 << " " << dm2 << endl;

	image<rgb_pixel> image(path);
	cout << "image in memory" << endl;

	size_t height = image.get_height(), width = image.get_width();

	cout << height << endl;
	cout << width << endl;
	cout << "image dim:" << height * width << endl;

	int* vec = new int[height * width];

	for (size_t y = 0; y < height; ++y) {
		for (size_t x = 0; x < width; ++x) {
			vec[y * height + x] = (int) image[y][x].red;
		}
	}

	cout << "convertion success" << endl;

	double* matr = calc_symmetric_adjacency_matrix(vec, width, height, dm1, dm2,
			max_i, max_j);
//	print_vector(matr, max_i, max_j);
	cout << "matrix ready" << endl;
	double *signs = calc_signs(matr, width, height, dm1, dm2, max_i, max_j);

	string noise_out = output_path + noise_path;
	mkdir(noise_out.c_str(), 0700);

	string out_data = noise_out + img_name + "_" + to_string(dm1) + "_" + to_string(dm2) + "_.txt";
	ofstream out(out_data.c_str());
	int signs_num = (int)signs[0];
	for(int j = 1; j<signs_num; j++){
		out << signs[j] << endl;
	}
	out.close();

	delete[] matr;
	delete[] vec;

	cout << "exec finished" << endl;
	return signs;
}

int main(int argc, char* argv[]) {

	Time start (boost::posix_time::microsec_clock::local_time());
	cout << "start time:" << start << endl;

	int dm1 = 5, dm2 = 5, max_i = 256, max_j = 256;

	if (argc == 3 || argc == 4) {
		dm1 = atoi(argv[1]);
		dm2 = atoi(argv[2]);
	}

	string noise_path; // this should be initialized
	if (argc == 4){
		noise_path = string(argv[3]);
	}

	string noise_out = output_path + noise_path + "output/";
	mkdir(noise_out.c_str(), 0700);

	cout << noise_out << endl;

	cout << path << endl;

	vector<string> *files = get_files_from_dir_with_noise(noise_path);

	double **signs = new double*[files->size()];

	int i = 0;

	for (vector<string>::iterator it = files->begin(); it != files->end();
			it++) {
		signs[i++] = exec(noise_path, path + noise_path + string(it->data()), string(it->data()), max_i,
				max_j, dm1, dm2);
	}

	int signs_num = (int)signs[0][0];

//	string noise_out = output_path + noise_path + "output/";
	mkdir(noise_out.c_str(), 0700);

	for (int i = 1; i < signs_num; i++) {
		string out_data = noise_out + to_string(i) + "_" + to_string(dm1) + "_" + to_string(dm2) + ".output.txt";
		ofstream out(out_data.c_str());
		for (int j = 0; j< files->size(); j++) {
			out << signs[j][i] << endl;
		}
		out.close();
	}


	for (int i = 0; i < files->size(); i++) {
		delete[] signs[i];
	}
	delete[] signs;
	delete files;

	Time end(boost::posix_time::microsec_clock::local_time());
	cout << "time spent:" << (end - start).total_milliseconds() << endl;

	return 0;
}
