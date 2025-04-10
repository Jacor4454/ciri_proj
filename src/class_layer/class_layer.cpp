#include "class_layer.h"

float inverse_sqrt(float number_){
    return 1/sqrt(number_);
}

std::default_random_engine BaseLayer::generator = std::default_random_engine();

std::vector<int> BaseLayer::makeWeightsDims(std::vector<int> in, std::vector<int> out){
    std::vector<int> output(in);
    int x = output.size();
    output[x-2] = output[x-1];
    output[x-1] = out[x-1];
    return output;
}

std::vector<int> BaseLayer::makeHWeightsDims(std::vector<int> out){
    std::vector<int> output(out);
    int x = output.size();
    output[x-2] = output[x-1];
    return output;
}

void BaseLayer::setAcc(activations::accTypes a){throw std::runtime_error("cannot ser acc type base layer");}
BaseLayer::~BaseLayer(){}

void BaseLayer::forward(tensor& output, const tensor& input){throw std::runtime_error("cannot forward base layer");}
void BaseLayer::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& output, const tensor&){throw std::runtime_error("cannot backward base layer");}
void BaseLayer::learn(){throw std::runtime_error("cannot learn base layer");}
void BaseLayer::clear(){throw std::runtime_error("cannot clear base layer");}

std::string BaseLayer::getLayerType(){return "Base";}
