#include "Layer.hpp"
#include "eigen/Eigen/Dense"


Layer::Layer(int size) {
    this->size = size;
    
    for(int i = 0; i < size; i++) {
        Neuron *n = new Neuron(0.000000000);
        this->neurons.push_back(n);
    }
}

Layer::Layer(int size, int activationType) {
    this->size = size;
    
    for(int i = 0; i < size; i++) {
        Neuron *n = new Neuron(0.000000000, activationType);
        this->neurons.push_back(n);
    }
}


vector<double> Layer::getActivatedVals() {
    vector<double> ret;
    
    for(int i = 0; i < this->neurons.size(); i++) {
        double v = this->neurons.at(i)->getActivatedVal();
        
        ret.push_back(v);
    }
    
    return ret;
}

void Layer::setVal(int i, double v) {
    this->neurons.at(i)->setVal(v);
}


void Layer::activateLayer(){
    for (int neuron_n = 0; neuron_n < this->neurons.size(); neuron_n++) {
        neurons.at(neuron_n)->activate();
    }
}


//EIGEN IMPLEMENTATIONS:
MatrixXd Layer::matrixifyVals() {
    MatrixXd m;
    m.resize(1, this->neurons.size());
    for(int i = 0; i < this->neurons.size(); i++) {
        m(0, i) = this->neurons.at(i)->getVal();
    }
    return m;
}

MatrixXd Layer::matrixifyActivatedVals() {
    MatrixXd m;
    m.resize(1, this->neurons.size());
    for(int i = 0; i < this->neurons.size(); i++) {
        m(0, i) = this->neurons.at(i)->getActivatedVal();
    }
    return m;
}

MatrixXd Layer::matrixifyDerivedVals() {
    MatrixXd m;
    m.resize(1, this->neurons.size());
    for(int i = 0; i < this->neurons.size(); i++) {
        m(0, i) = this->neurons.at(i)->getDerivedVal();
    }
    return m;
}




