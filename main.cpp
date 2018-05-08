//
//  main.cpp
//  NN
//
//  Created by preston sundar on 4/26/18.
//  Copyright © 2018 prestonsundar. All rights reserved.
//

//TIME WITHOUT EIGEN ≈ 1.11-1.4s

#include <iostream>
#include "NeuralNetwork.hpp"
#include <cstdio>
#include <ctime>
#include "eigen/Eigen/Dense"

using namespace Eigen;
using namespace std;

MatrixXd vectToMatrixXd_colWise(vector<double> _vect){
    MatrixXd ret;
    ret.resize(1, _vect.size());
    for (unsigned index = 0; index < _vect.size(); index++) {
        ret(0, index) = _vect.at(index);
    }
    return ret;
}

void printVect(vector<double> vect_double){
    for (auto iter = vect_double.begin(); iter != vect_double.end(); iter++) {
        cout << *iter << '\t';
    }
}

int main(int argc, const char * argv[]) {
//    std::clock_t start;
//    double duration;
//
//    MatrixXd input, target;
//    input.resize(1, 4);
//    target.resize(1, 4);
//    target << 0.5, 0.1, 0.9, 0.2;
//
//
//    vector<MatrixXd> trainingMatrices;
//    trainingMatrices.resize(4);
//
//    for (unsigned i = 0; i < trainingMatrices.size(); i++) {
//        trainingMatrices.at(i).resize(1, 3);
//        cout << i;
//    }
//
//
//
//
//    NeuralNetwork *nn = new NeuralNetwork({2, 5, 1}, 0.05, 1);
//    double err = 1;
//    unsigned long iter_n = 0;
//
//    start = std::clock();
//
//    while(iter_n < 500) {
//        iter_n++;
//        nn->train(input, target, 0.05, 1);
//        err = nn->getCurrentError();
//
//        if (iter_n % 100 == 0) {
//            cout << "Training at index: " << iter_n << endl;
//            cout << "Error: " << err << endl;
//        }
//
//
//    }
//
//    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
//
//
//    cout << "Training Complete with final error: " << nn->getCurrentError() << " | Time taken: " << duration << "s";
//    cout << " | Iterations: " << iter_n << endl;
//    cout << "testing with input values. Output is: ";
//
//    vector<double> outputs = nn->getNeuralNetOutputs(input);
//
//    for (auto outN = outputs.begin(); outN != outputs.end(); outN++) {
//        cout << *outN << " ";
//    }
//
//    cout << endl;
//
    
    std::clock_t start;
    double duration;

    MatrixXd input, target, test;
    
    input.resize(1, 2);
    input << 0.5, 0.1;

    target.resize(1, 5);
    target << 0.8, 0.45, 0.1, 0.3, 0.9;
    
    MatrixXd input2, target2;
    
    input2.resize(1, 2);
    input2 << 0.9999, 0.3;
    
    target2.resize(1, 5);
    target2 << 0.1, 0.1, 0.1, 0.1, 0.1;
    
    double learingRate = 0.05;
    double momentum = 0.99;
    double totalError = 0;
    
    
    NeuralNetwork *nn = new NeuralNetwork({2, 5, 10, 5}, learingRate, momentum);

    start = std::clock();
    
    
    cout << nn->getBiasMatrix(2);
    cout << "\n";
    
    for (unsigned cycle_n = 0; cycle_n < 10000; cycle_n++) {
        
        nn->train(input, target, learingRate, momentum);
        //nn->train(input2, target2, learingRate, momentum);
        
        if (cycle_n % 100 == 0) {
            cout << "Iteration: " << cycle_n << endl;
            cout << "Current error: " << nn->getCurrentError() << endl;
            cout << "Current output 1: ";
            printVect(nn->getNeuralNetOutputs(input));
            cout << "\n\n===============================";
//            cout << "Current output 2: ";
//            printVect(nn->getNeuralNetOutputs(input2));
//            cout << "\n\n\n";
//
        }
            

        
    }
    
    cout << nn->getBiasMatrix(2);
    cout << "\n";
    
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    
    
    cout << "Training Complete with final error: " << totalError << " |Time taken: " << duration << "s" << endl;
    cout << "testing with input values. Output is: ";
    

    printVect(nn->getNeuralNetOutputs(vectToMatrixXd_colWise({0.7, -0.5})));
    
    cout << endl;
    
    
    
    return 0;
}
