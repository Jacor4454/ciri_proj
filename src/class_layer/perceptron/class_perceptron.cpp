#include "class_perceptron.h"


perceptron::perceptron(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    weights(weightsDims),
    bias(out),
    hold(out)
{
    dweight = new BaseLearner(&weights, 0.1);
    dbias = new BaseLearner(&bias, 0.1);
}

perceptron::~perceptron(){
    delete dweight;
    delete dbias;
}

void perceptron::forward(tensor& output, const tensor& input){
    input.addAndMult(output, weights, bias);

    output.activate(activations::ReLU);
}

void perceptron::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& IGNORE, tensor& output){
    // deactivate output
    output.deactivate(activations::ReLU);
    
    // smultiply deactivated output, dOutput and dInt
    dOutput.sMult(hold, output);

    // get dweights
    dweight->backprop(input, hold);

    // get dbias
    dbias->backprop(hold);

    // get dinput and dInternal carry
    hold.fastDeMultR(dInput, weights);
}

void perceptron::learn(){
    dweight->learn();
    dbias->learn();
}

std::string perceptron::getLayerType(){return "Perceptron";}
