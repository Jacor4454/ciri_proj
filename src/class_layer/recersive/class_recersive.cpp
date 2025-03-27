#include "class_recersive.h"

recursive::recursive(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    hweightsDims(BaseLayer::makeHWeightsDims(out)),
    weights(weightsDims),
    bias(out),
    rWeights(hweightsDims),
    prev(out),
    dInt(out),
    dbias(out),
    dweights(weightsDims),
    dweightsTemp(weightsDims),
    dRWeights(hweightsDims),
    dRWeightsTemp(hweightsDims)
{}

recursive::~recursive(){}

void recursive::forward(tensor& output, const tensor& input){
    input.addAndMult(output, weights, bias);
    prev.multAndInc(output, rWeights);
    
    prev.cpy(output);
    
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
