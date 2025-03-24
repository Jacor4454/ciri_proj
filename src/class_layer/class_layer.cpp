#include "class_layer.h"

BaseLayer::~BaseLayer(){}

void BaseLayer::forward(tensor& output, const tensor& input){throw std::runtime_error("cannot forward base layer");}
void BaseLayer::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& output, tensor&){throw std::runtime_error("cannot backward base layer");}
void BaseLayer::learn(float alpha){throw std::runtime_error("cannot learn base layer");}

std::vector<int> BaseLayer::getDims(){return dims;}

std::string BaseLayer::getLayerType(){return "Base";}
