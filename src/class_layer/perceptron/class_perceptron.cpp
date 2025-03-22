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
    std::cout << "1\n";
    input.mult(dweightsTemp, dOutput);
    std::cout << "a\n";
    dweights.add(dweights, dweightsTemp);
    std::cout << "b\n";

    dbias.add(dbias, dOutput);
    std::cout << "c\n";

    weights.mult(dInput, dOutput);
    std::cout << "d\n";
}

std::string perceptron::getLayerType(){return "Perceptron";}
