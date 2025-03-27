#include "class_layer.h"


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

BaseLayer::~BaseLayer(){}

void BaseLayer::forward(tensor& output, const tensor& input){throw std::runtime_error("cannot forward base layer");}
void BaseLayer::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& output, tensor&){throw std::runtime_error("cannot backward base layer");}
void BaseLayer::learn(float alpha){throw std::runtime_error("cannot learn base layer");}

std::string BaseLayer::getLayerType(){return "Base";}
