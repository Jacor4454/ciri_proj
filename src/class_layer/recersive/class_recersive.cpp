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
    dweight = new BaseLearner(&weights, 0.02);
    drweight = new BaseLearner(&rWeights, 0.02);
    dbias = new BaseLearner(&bias, 0.02);
    weights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(weights.getN()), inverse_sqrt(weights.getN()));
    rWeights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(rWeights.getN()), inverse_sqrt(rWeights.getN()));
    bias.sMult(bias, 0);
    acc = activations::ReLU;
}

void recursive::setAcc(activations::accTypes a){acc = a;}

recursive::~recursive(){
    delete dweight;
    delete drweight;
    delete dbias;
}

void recursive::forward(tensor& output, const tensor& input){
    input.addAndMult(output, weights, bias);
    prev.multAndInc(output, rWeights);
    
    prev.cpy(output);
    
    output.activate(acc);
}

void recursive::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& prevoutput, tensor& output){
    // deactivate output
    output.deactivate(acc);

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

void recursive::clear(){
    dweight->clear();
    drweight->clear();
    dbias->clear();
    prev.sMult(prev, 0.0);// for now
    dInt.sMult(dInt, 0.0);// for now
}

std::string recursive::getLayerType(){return "Recersive";}
