#ifndef _NEURON_HPP_
#define _NEURON_HPP_

#define TANH 1
#define RELU 2
#define SIGM 3

#include <iostream>
#include <math.h>
using namespace std;

class Neuron
{
public:
    Neuron(double val);
    Neuron(double val, int activationType);
    
    void setVal(double v);
    
    void activate();
    
    void derive();
    
    // Getter
    double getVal() { return this->val; }
    double getActivatedVal() { return this->activatedVal; }
    double getDerivedVal() { return this->derivedVal; }
    
private:
    double val;
    double activatedVal;
    double derivedVal;
    double bias;
    
    int activationType = 3;
    
    double sigmoid(double x){
        return 1 / (1 + exp(-x));
    }
    
    double d_sigmoid(double x){
        return this->sigmoid(x) * (1 - this->sigmoid(x));
    }
};

#endif
