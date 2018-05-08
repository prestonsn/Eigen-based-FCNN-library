#ifndef _NEURAL_NETWORK_HPP_
#define _NEURAL_NETWORK_HPP_

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <time.h>
//#include "Matrix.hpp"
#include "Layer.hpp"
#include "eigen/Eigen/Dense"



enum NN_COST {
    COST_MSE
};

enum NN_ACTIVATION {
    A_TANH,
    A_RELU,
    A_SIGM
};

class NeuralNetwork{
public:
    NeuralNetwork(std::vector<unsigned> Topology, double learningRate, double momentum);
    
    void setCurrentInput(Eigen::MatrixXd input);
    void setCurrentTarget(Eigen::MatrixXd target) { this->currentTarget = target; }
    
    void feedForward();
    void backPropagation();
    void setErrors();
    void train(Eigen::MatrixXd input, Eigen::MatrixXd target, double learningRate, double momentum);
    void trainDataSet(std::vector<Eigen::MatrixXd> inputMats, std::vector<Eigen::MatrixXd> targetMats, double learningRate, double momentum);
  
    
    std::vector<double> getNeuralNetOutputs(Eigen::MatrixXd Inputs);
    std::vector<double> getActivatedVals(int index) { return this->layers.at(index)->getActivatedVals(); }
    
    Eigen::MatrixXd getNeuronMatrix(int index)          { return this->layers.at(index)->matrixifyVals(); }
    Eigen::MatrixXd getActivatedNeuronMatrix(int index) { return this->layers.at(index)->matrixifyActivatedVals(); }
    Eigen::MatrixXd getDerivedNeuronMatrix(int index)   { return this->layers.at(index)->matrixifyDerivedVals(); }
    Eigen::MatrixXd getWeightMatrix(int index)          { return this->weightMatrices.at(index); }
    Eigen::MatrixXd getBiasMatrix(int index)            { return this->biasMatrices.at(index); }
    
    void setNeuronValue(int indexLayer, int indexNeuron, double val) { this->layers.at(indexLayer)->setVal(indexNeuron, val); }
    
    unsigned long topologySize;
    int hiddenActivationType  = RELU;
    int outputActivationType  = SIGM;
    int costFunctionType      = COST_MSE;
    
    std::vector<unsigned> topology;
    std::vector<Layer *> layers;
    std::vector<Eigen::MatrixXd> weightMatrices;
    std::vector<Eigen::MatrixXd> gradientMatrices;
    std::vector<Eigen::MatrixXd> biasMatrices;

    
    Eigen::MatrixXd currentInput;
    Eigen::MatrixXd currentTarget;
    Eigen::MatrixXd outputErrors;
    Eigen::MatrixXd outputErrors_derivatives;
    
    double currentError       = 0;
    std::vector<double> trainingSetErrors;
    
    double momentum;
    double learningRate;
    double regularization;
    
    double getCurrentError(void){
        return this->currentError;
    }
    
    void updateWeightMatrix(int index, Eigen::MatrixXd dE_dW){
        this->weightMatrices.at(index) = (this->momentum * this->weightMatrices.at(index)) + (this->learningRate * dE_dW);
        //cout << "\nWeight mat \n" << index << endl << this->weightMatrices.at(index);
    }
    
    void updateBiasMatrix(int index, Eigen::MatrixXd dE_dB){
        this->biasMatrices.at(index) -= (this->learningRate/10) * dE_dB;
    }
    
    
private:
    void setErrorMSE();
};

#endif
