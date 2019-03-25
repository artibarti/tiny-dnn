/*
	Copyright (c) 2013, Taiga Nomi and the respective contributors
	All rights reserved.
	
	Use of this source code is governed by a BSD-style license that can be found
	in the LICENSE file.
*/

// this example shows how to use tiny-dnn library to fit data with OpenCL 
// support, by learning a sinus function.
// please also see: https://github.com/tiny-dnn/tiny-dnn/blob/master/docs/how_tos/How-Tos.md

#include <iostream>
#include "tiny_dnn/tiny_dnn.h"

void train(tiny_dnn::network<tiny_dnn::sequential>& net) {

	std::vector<tiny_dnn::vec_t> X;
	std::vector<tiny_dnn::vec_t> sinusX;

	// training data
	for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
		tiny_dnn::vec_t vx = {x};
		tiny_dnn::vec_t vsinx = {sinf(x)};
		X.push_back(vx);
		sinusX.push_back(vsinx);
	}

	// learning parameters 
	// (16 samples for each weight update /2000 presentations of all samples)
	size_t batch_size = 16;
	int epochs = 2000;
	tiny_dnn::adamax opt;

	// epoch callback
	int iEpoch = 0;
	auto runAfterEpoch = [&]() {
		
		iEpoch++;
		if (iEpoch % 100) {
			return; 
		}
		
		double loss = net.get_loss<tiny_dnn::mse>(X, sinusX);
		std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss << std::endl;
	};

	std::cout << "learning the sinus function with 2000 epochs:" << std::endl;
	net.fit<tiny_dnn::mse>(opt, X, sinusX, batch_size, 
		epochs, []() {}, runAfterEpoch);

	std::cout << std::endl << "Training finished" << std::endl << std::endl;
}

void test(tiny_dnn::network<tiny_dnn::sequential>& net) {

	float fMaxError = 0.f;
	for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
		
		tiny_dnn::vec_t xv = {x};
		float fPredicted = net.predict(xv)[0];
		float fDesired = sinf(x);

		std::cout << "x=" << x << " sinX=" << fDesired
			<< " predicted=" << fPredicted << std::endl;

		float fError = fabs(fPredicted - fDesired);

		if (fMaxError < fError) 
			fMaxError = fError;
	}
	std::cout << std::endl << "max_error=" << fMaxError << std::endl << std::endl;

}

void initNetwork(tiny_dnn::network<tiny_dnn::sequential>& net, tiny_dnn::core::backend_t backend) {

	net << tiny_dnn::fully_connected_layer(1, 10, false, backend);
	net << tiny_dnn::tanh_layer();
	net << tiny_dnn::fully_connected_layer(10, 10, false, backend);
	net << tiny_dnn::tanh_layer();
	net << tiny_dnn::fully_connected_layer(10, 1, false, backend);
}

int main() {

	tiny_dnn::network<tiny_dnn::sequential> net_internal;
	initNetwork(net_internal, tiny_dnn::core::backend_t::internal);
	
	std::cout << "Train net with internal backend..." << std::endl;
	train(net_internal);

	std::cout << "Test net with internal backend..." << std::endl;
	test(net_internal);

	// some weird thing is going on here, the previous network has an effect on the second one,
	// instead of having the same output, the second one gives a more precise result

	tiny_dnn::network<tiny_dnn::sequential> net_opencl;
	initNetwork(net_opencl, tiny_dnn::core::backend_t::opencl);

	std::cout << "Train net with opencl backend..." << std::endl;
	train(net_opencl);

	std::cout << "Test net with opencl backend..." << std::endl;
	test(net_opencl);

	return 0;
}
