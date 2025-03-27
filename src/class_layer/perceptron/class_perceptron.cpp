#include "class_perceptron.h"


perceptron::perceptron(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    weights(weightsDims),
    bias(out),
    dweights(weightsDims),
    dweightsTemp(weightsDims),
    dbias(out),
    hold(out)
{}

perceptron::~perceptron(){}

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
