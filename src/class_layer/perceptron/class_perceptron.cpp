#include "class_perceptron.h"

perceptron::perceptron(int in, int out):
    weights({in,out}),
    bias({1,out}),
    dweights({in,out}),
    dweightsTemp({in,out}),
    dbias({1,out}),
    hold({1,out})
{}

perceptron::~perceptron(){}

void perceptron::forward(tensor& output, const tensor& input){
    input.mult(output, weights);
    output.add(output, bias);

    output.activate(activations::ReLU);
}

void perceptron::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& IGNORE, tensor& output){
    // deactivate output
    output.deactivate(activations::ReLU);
    
    // smultiply deactivated output, dOutput and dInt
    dOutput.sMult(hold, output);

    // get dweights
    input.fastDeMultL(dweightsTemp, dOutput);
    dweights.add(dweights, dweightsTemp);

    // get dbias
    dbias.add(dbias, dOutput);

    // get dinput and dInternal carry
    dOutput.fastDeMultR(dInput, weights);
}

void perceptron::learn(float alpha){
    dbias.sMult(dbias, -alpha);
    bias.add(bias, dbias);

    dweights.sMult(dweights, -alpha);
    weights.add(weights, dweights);
}

std::string perceptron::getLayerType(){return "Perceptron";}
