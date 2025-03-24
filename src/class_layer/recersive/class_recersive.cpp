#include "class_recersive.h"

recursive::recursive(int in, int out):
    weights({in,out}),
    bias({1,out}),
    rWeights({out,out}),
    prev({1,out}),
    dInt({1,out}),
    forwardTemp({1,out}),
    dbias({1,out}),
    dweights({in,out}),
    dweightsTemp({in,out}),
    dRWeights({out,out}),
    dRWeightsTemp({out,out})
{}

recursive::~recursive(){}

void recursive::forward(tensor& output, const tensor& input){
    input.mult(output, weights);
    
    prev.mult(forwardTemp, rWeights);
    prev.cpy(input);

    output.add(output, forwardTemp);
    output.add(output, bias);
    
    output.activate(activations::ReLU);
}

void recursive::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& prevoutput, tensor& output){
    // deactivate output
    output.deactivate(activations::ReLU);

    // smultiply deactivated output, dOutput and dInt
    dInt.sMult(dInt, dOutput);
    dInt.sMult(dInt, output);

    // get dweights
    input.fastDeMultL(dweightsTemp, dInt);
    dweights.add(dweights, dweightsTemp);

    // get dbias
    dbias.add(dbias, dInt);

    // get drweigths
    prevoutput.fastDeMultL(dRWeightsTemp, dInt);
    dRWeights.add(dRWeights, dRWeightsTemp);

    // get dinput and dInternal carry
    dInt.fastDeMultR(dInput, weights);
    dInt.fastDeMultR(dInt, rWeights);
}

void recursive::learn(float alpha){
    dbias.sMult(dbias, -alpha);
    bias.add(bias, dbias);

    dweights.sMult(dweights, -alpha);
    weights.add(weights, dweights);

    dRWeights.sMult(dRWeights, -alpha);
    rWeights.add(rWeights, dRWeights);
}

std::string recursive::getLayerType(){return "Recersive";}
