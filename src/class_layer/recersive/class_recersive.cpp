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
    dweight = nullptr;
    drweight = nullptr;
    dbias = nullptr;
    weights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(weights.getN()), inverse_sqrt(weights.getN()));
    rWeights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(rWeights.getN()), inverse_sqrt(rWeights.getN()));
    bias.set(0.0);
    acc = activations::ReLU;
}

void recursive::setAcc(activations::accTypes a){acc = a;}

recursive::~recursive(){
    // delete heap memory
    if(dweight != nullptr)
        delete dweight;
    if(drweight != nullptr)
        delete drweight;
    if(dbias != nullptr)
        delete dbias;
}

void recursive::forward(tensor& output, const tensor& input){

    // do mx+c
    input.addAndMult(output, weights, bias);

    // add mx
    prev.multAndInc(output, rWeights);
    
    // activate
    output.activate(acc);

    // store for next itt
    prev.cpy(output);
}

void recursive::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& prevoutput, const tensor& output){
    // allocate learning tensors if needed
    if(dweight == nullptr)
        dweight = new BaseLearner(&weights, 0.1);
    if(drweight == nullptr)
        drweight = new BaseLearner(&rWeights, 0.1);
    if(dbias == nullptr)
        dbias = new BaseLearner(&bias, 0.1);

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
    if(dweight != nullptr)
        dweight->learn();
    if(drweight != nullptr)
        drweight->learn();
    if(dbias != nullptr)
        dbias->learn();
}

void recursive::clear(){
    if(dweight != nullptr)
        dweight->clear();
    if(drweight != nullptr)
        drweight->clear();
    if(dbias != nullptr)
        dbias->clear();
    prev.set(0.0);
    dInt.set(0.0);
}

std::string recursive::getLayerType(){return "Recersive";}
