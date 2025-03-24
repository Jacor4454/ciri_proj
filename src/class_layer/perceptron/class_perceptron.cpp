#include "class_perceptron.h"

perceptron::perceptron(int in, int out):
    weights({in,out}),
    bias({1,out}),
    dweights({in,out}),
    dweightsTemp({in,out}),
    dbias({1,out})
{}

perceptron::~perceptron(){}

void perceptron::forward(tensor& output, const tensor& input){
    input.mult(output, weights);
    output.add(output, bias);
}

void perceptron::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& IGNORE){
    input.fastDeMultL(dweightsTemp, dOutput);
    dweights.add(dweights, dweightsTemp);

    dbias.add(dbias, dOutput);

    dOutput.fastDeMultR(dInput, weights);
}

std::string perceptron::getLayerType(){return "Perceptron";}
