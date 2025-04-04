#include "class_recersive.h"

recursive::recursive(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    hweightsDims(BaseLayer::makeHWeightsDims(out)),
    weights(weightsDims),
    bias(out),
    rWeights(hweightsDims),
    prev(out),
    dInt(out)
{
    dweight = new BaseLearner(&weights, 0.1);
    drweight = new BaseLearner(&rWeights, 0.1);
    dbias = new BaseLearner(&bias, 0.1);
}

recursive::~recursive(){
    delete dweight;
    delete drweight;
    delete dbias;
}

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
    dweight->backprop(input, dInt);

    // get dbias
    dbias->backprop(dInt);

    // get drweigths
    drweight->backprop(prevoutput, dInt);

    // get dinput and dInternal carry
    dInt.fastDeMultR(dInput, weights);
    dInt.fastDeMultR(dInt, rWeights);
}

void recursive::learn(){
    dweight->learn();
    drweight->learn();
    dbias->learn();
}

std::string recursive::getLayerType(){return "Recersive";}
