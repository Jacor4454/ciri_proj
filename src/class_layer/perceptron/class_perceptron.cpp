#include "class_perceptron.h"

perceptron::perceptron(std::vector<int> in, std::vector<int> out):
    weightsDims(BaseLayer::makeWeightsDims(in, out)),
    weights(weightsDims),
    bias(out),
    hold(out)
{   
    // we want to only define if learning yk
    dweight = nullptr;
    dbias = nullptr;
    weights.xavierRnd(BaseLayer::generator, -1*inverse_sqrt(weights.getN()), inverse_sqrt(weights.getN()));
    bias.set(0.0);
    acc = activations::ReLU;
    bls = new AlphaLearnerSelector(0.01);
}

void perceptron::setAcc(activations::accTypes a){acc = a;}

void perceptron::setLearners(BaseLearnerSelector* bls_){
    if(bls_ == nullptr)
        return;
    
    if(dweight != nullptr)
        delete dweight;
    if(dbias != nullptr)
        delete dbias;
    if(bls != nullptr)
        delete bls;

    bls = bls_;
    dweight = bls->construct(&weights);
    dbias = bls->construct(&bias);
}

perceptron::~perceptron(){
    // delete heap memory
    if(dweight != nullptr)
        delete dweight;
    if(dbias != nullptr)
        delete dbias;
}

void perceptron::forward(tensor& output, const tensor& input){
    // do mx+c
    input.addAndMult(output, weights, bias);

    // activate
    output.activate(acc);
}

void perceptron::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& IGNORE, const tensor& output){
    // allocate learning tensors if needed
    if(dweight == nullptr)
        dweight = bls->construct(&weights);
    if(dbias == nullptr)
        dbias = bls->construct(&bias);

    // deactivate output
    output.deactivate(hold, acc);
    
    // smultiply deactivated output, dOutput and dInt
    hold.sMult(hold, dOutput);

    // get dweights
    dweight->backprop(input, hold);

    // get dbias
    dbias->backprop(hold);

    // get dinput and dInternal carry
    hold.fastDeMultR(dInput, weights);
}

void perceptron::learn(){
    if(dweight != nullptr)
        dweight->learn();
    if(dbias != nullptr)
        dbias->learn();
}

void perceptron::clear(){
    if(dweight != nullptr)
        dweight->clear();
    if(dbias != nullptr)
        dbias->clear();
}

std::string perceptron::getLayerType(){return "Perceptron";}
