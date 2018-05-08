#ifndef _LAYER_HPP_
#define _LAYER_HPP_

#include <iostream>
#include "Neuron.hpp"
#include "Matrix.hpp"
#include "eigen/Eigen/Dense"

using namespace Eigen;

class Layer
{
public:
    Layer(int size);
    Layer(int size, int activationType);
    void setVal(int i, double v);
    void activateLayer();
    
    
    MatrixXd matrixifyVals();
    MatrixXd matrixifyActivatedVals();
    MatrixXd matrixifyDerivedVals();
    
    vector<double> getActivatedVals();
    
    vector<Neuron *> getNeurons() { return this->neurons; };
    void setNeuron(vector<Neuron *> neurons) { this->neurons = neurons; }
private:
    int size;
    vector<Neuron *> neurons;
};

#endif
