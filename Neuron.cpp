#include "Neuron.hpp"

// Constructor
Neuron::Neuron(double val) {
    this->setVal(val);
}

Neuron::Neuron(double val, int activationType) {
    this->activationType = activationType;
    this->setVal(val);
   
}

void Neuron::setVal(double val) {
    this->val = val;
    activate();
    derive();
}

void Neuron::activate() {
    
    if(activationType == TANH) {
        this->activatedVal = tanh(this->val);
        //cout << "tanh" << endl;
    } else if(activationType == RELU) {
        //cout << "RELU" << endl;
        if(this->val > 0) {
            this->activatedVal = this->val;
        } else {
            this->activatedVal = 0;
        }
    } else if(activationType == SIGM) {
        //cout << "SIGM" << endl;
        this->activatedVal = sigmoid(this->val);
    } else {
        //cout << "SIGM" << endl;
        this->activatedVal = sigmoid(this->val);
    }
    
}

void Neuron::derive() {
    
    if(activationType == TANH) {
        this->derivedVal = (1.0 - (tanh(this->val) * tanh(this->val)));
    } else if(activationType == RELU) {
        if(this->val > 0) {
            this->derivedVal = 1;
        } else {
            this->derivedVal = 0;
        }
    } else if(activationType == SIGM) {
        this->derivedVal = this->d_sigmoid(this->val);
    } else {
        this->derivedVal = this->d_sigmoid(this->val);
    }
    
}





