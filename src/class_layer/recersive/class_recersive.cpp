#include "class_recersive.h"

recursive::recursive(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    hweightsDims(BaseLayer::makeHWeightsDims(out)),
    weights(weightsDims),
    bias(out),
    rWeights(hweightsDims),
    prev(out),
    dInt(out),
    hold(out)
{
    dweight = new BaseLearner(&weights, 0.1);
    drweight = new BaseLearner(&rWeights, 0.1);
    dbias = new BaseLearner(&bias, 0.1);
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
    // std::cout << "\nrecForward\n";
    input.addAndMult(output, weights, bias);
    // std::cout << weights << bias << input << output;
    prev.multAndInc(output, rWeights);
    // std::cout << prev << rWeights << output;
    
    output.activate(acc);

    prev.cpy(output);
    // std::cout << output;
    // std::cout << "recForwardEnd\n\n";
}

void recursive::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& prevoutput, const tensor& output){
    // deactivate output
    output.deactivate(hold, acc);

    // smultiply deactivated output, dOutput and dInt
    dInt.add(dInt, dOutput);
    hold.sMult(hold, dInt);

    // get dweights
    dweight->backprop(input, hold);

    // get dbias
    dbias->backprop(hold);

    // get drweigths
    drweight->backprop(prevoutput, hold);

    // get dinput and dInternal carry
    hold.fastDeMultR(dInput, weights);
    hold.fastDeMultR(dInt, rWeights);
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
