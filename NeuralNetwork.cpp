#include "NeuralNetwork.hpp"
#include "json.hpp"

#include "NeuralNetwork.hpp"

// Constructor
NeuralNetwork::NeuralNetwork(vector<unsigned> Topology, double learningRate, double momentum) {
    //this->config        = config;
    this->topology      = Topology;
    this->topologySize  = Topology.size();
    this->learningRate  = learningRate;
    this->momentum      = momentum;
    //this->bias          = bias;
    
    this->hiddenActivationType  = SIGM;
    this->outputActivationType  = SIGM;
    this->costFunctionType      = COST_MSE;
    
    for(int i = 0; i < topologySize; i++) {
        if(i > 0 && i < (topologySize - 1)) {
            Layer *l  = new Layer(topology.at(i), this->hiddenActivationType);
            this->layers.push_back(l);
        } else if(i == (topologySize - 1)) {
            Layer *l  = new Layer(topology.at(i), this->outputActivationType);
            this->layers.push_back(l);
        } else {
            Layer *l  = new Layer(topology.at(i));
            this->layers.push_back(l);
        }
    }
    
    for(int i = 0; i < (topologySize - 1); i++) {
        //Matrix *m = new Matrix(topology.at(i), topology.at(i + 1), true);
        MatrixXd w;
        w.resize(topology.at(i), topology.at(i + 1));
        std::srand((unsigned int) time(0));
        w.setRandom();
        this->weightMatrices.push_back(w);
    }
    
    this->outputErrors_derivatives.resize(1, this->topology.at(this->topologySize - 1));
    this->outputErrors.resize(1, this->topology.at(this->topologySize - 1));
    
    this->currentError = 0.00;
    
    //set up the bias matrices
    for (int i = 0; i < topologySize - 1; i++) {
        MatrixXd b;
        b.resize(this->getNeuronMatrix(i).rows(), this->getWeightMatrix(i).cols());
        b.setOnes();
        this->biasMatrices.push_back(b);
    }
}


void NeuralNetwork::feedForward() {
    MatrixXd layerVals;  // Matrix of neurons to the left
    MatrixXd layerWeights;  // Matrix of weights to the right of layer
    MatrixXd nextLayerVals;  // Matrix of neurons to the next layer
    MatrixXd layerBias;
    
    
    for(int i = 0; i < (this->topologySize - 1); i++) {
        layerVals = this->getNeuronMatrix(i);
        layerWeights = this->getWeightMatrix(i);
        layerBias = this->getBiasMatrix(i);
        
        nextLayerVals.resize(layerVals.rows(), layerWeights.cols());
        
        if(i != 0) {
            layerVals = this->getActivatedNeuronMatrix(i);
        }else{
            //cout << "FF with: " << layerVals << endl;
        }
        
        nextLayerVals = (layerVals * layerWeights) + layerBias;
        
        for(int c_index = 0; c_index < nextLayerVals.cols(); c_index++) {
            this->setNeuronValue(i + 1, c_index, nextLayerVals(0, c_index));
        }
        
    }
    
    this->trainingSetErrors.resize(10);
    
}


void NeuralNetwork::setErrors() {
    switch(costFunctionType) {
        case(COST_MSE): this->setErrorMSE(); break;
        default: this->setErrorMSE(); break;
    }
}

void NeuralNetwork::setErrorMSE() {
    Eigen::MatrixXd Target = this->currentTarget;
    Eigen::MatrixXd Outputs = this->layers.at(this->topologySize - 1)->matrixifyActivatedVals();
    
    Eigen::MatrixXd difference = Target - Outputs;
    
    this->outputErrors_derivatives = -difference;
    this->outputErrors = 0.5 * difference.array().square();//raise to power of two
    
    
    this->currentError = this->outputErrors.sum();
}


//void NeuralNetwork::setError_external(double Error, double ErrorDerivatives) {
//
//    
//    this->outputErrors_derivatives = ErrorDerivatives;
//    this->outputErrors = 0.5 * difference.array().square();//raise to power of two
//    
//    
//    this->currentError = this->outputErrors.sum();
//}





void NeuralNetwork::backPropagation() {
    Eigen::MatrixXd dE_dW;
    Eigen::MatrixXd dE_dB;
    Eigen::MatrixXd currentDelta;
    Eigen::MatrixXd prevDelta;
    Eigen::MatrixXd testMat;
    
    int indexOutputLayer = this->topology.size() - 1;
    
    //get delta
    currentDelta = this->outputErrors_derivatives;
    currentDelta.cwiseProduct(this->getDerivedNeuronMatrix(indexOutputLayer));
    
    //get dE_dW;
    dE_dW = currentDelta.transpose() * this->getActivatedNeuronMatrix(indexOutputLayer - 1);
    
    //get dE_dB;
    dE_dB = currentDelta;
    
    //cout << "BP test point 1: output layer gradients done" << endl;
    
    //update our parameters(stochastic)
    this->updateBiasMatrix(indexOutputLayer - 1, dE_dB);
    this->updateWeightMatrix(indexOutputLayer - 1, dE_dW.transpose());
    
    //save delta to be used in next layer adjustments
    prevDelta = currentDelta;
    
    //cout << "BP test point 2: output weight and Bias succesful" << endl;
    
    //now repeat process for remaining layers
    for (int layer_n = indexOutputLayer - 1; layer_n > 1; layer_n--) {
        //calculate the current layer's new delta
        //currentDelta = (nextlayerWeights * prevDelta) o currentLayerderivatives
        //testMat = this->getWeightMatrix(layer_n);
        currentDelta = prevDelta * this->getWeightMatrix(layer_n).transpose();
        currentDelta.cwiseProduct(this->getDerivedNeuronMatrix(layer_n));
        
        //calculate dE_dW and dE_dB using the delta * prevLayerOutput
        //testMat =this->getActivatedNeuronMatrix(layer_n - 1);
        dE_dW = currentDelta.transpose() * this->getActivatedNeuronMatrix(layer_n - 1);
        dE_dB = currentDelta;
        
        //update parameters
        this->updateWeightMatrix(layer_n - 1, dE_dW.transpose());
        this->updateBiasMatrix(layer_n - 1, dE_dB);
        
        //save the current delta
        prevDelta = currentDelta;
        
        //cout << "BP test point 3: layer " << layer_n << " update done" << endl;
        
    }
    
    //repeat for input layer
    
    //calculate the input layer's new delta
    //currentDelta = (nextlayerWeights * prevDelta) o currentLayerderivatives
    
    currentDelta = prevDelta * this->getWeightMatrix(1).transpose();
    testMat = this->getDerivedNeuronMatrix(0);
    currentDelta.cwiseProduct(this->getDerivedNeuronMatrix(1));
    
    //calculate dE_dW and dE_dB using the delta * inputs
    dE_dW = currentDelta.transpose() * this->getNeuronMatrix(0);
    dE_dB = currentDelta;
    
    //update parameters
    this->updateWeightMatrix(0, dE_dW.transpose());
    this->updateBiasMatrix(0, dE_dB);
}

void NeuralNetwork::train(
                          Eigen::MatrixXd input,
                          Eigen::MatrixXd target,
                          double learningRate,
                          double momentum
                          ) {
    
    
    this->learningRate  = learningRate;
    this->momentum      = momentum;
    
    this->setCurrentInput(input);
    this->setCurrentTarget(target);
    

    this->feedForward();

    this->setErrors();

    this->backPropagation();

}

//void NeuralNetwork::trainDataSet(std::vector<Eigen::MatrixXd> inputMats, std::vector<Eigen::MatrixXd> targetMats, double learningRate, double momentum){
//
//
//
//
//
//    this->learningRate  = learningRate;
//    this->momentum      = momentum;
//
//    this->setCurrentInput(input);
//    this->setCurrentTarget(target);
//
//    
//    this->feedForward();
//
//    this->setErrors();
//
//    this->backPropagation();
//
//
//    for (unsigned _targetNum = 0; _targetNum < targetMats.size(); _targetNum++) {
//        this->setCurrentInput(inputMats.at(_targetNum));
//        this->setCurrentTarget(targetMats.at(_targetNum));
//        this->feedForward();
//
//
//
//        this->backPropagation();
//
//    }
//
//
//
//
//
//
//}

//void NeuralNetwork::setCustomErrors()



void NeuralNetwork::setCurrentInput(Eigen::MatrixXd input) {
    this->currentInput = input;
    
    for(int i = 0; i < input.cols(); i++) {
        this->layers.at(0)->setVal(i, input(0, i));
    }
}

vector<double> NeuralNetwork::getNeuralNetOutputs(Eigen::MatrixXd Inputs){
    this->setCurrentInput(Inputs);
    feedForward();
    this->layers.at(this->topologySize - 1)->activateLayer();
    return this->layers.at(this->topologySize - 1)->getActivatedVals();
}
