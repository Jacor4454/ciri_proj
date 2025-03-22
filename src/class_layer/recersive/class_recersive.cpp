#include "class_recersive.h"

recursive::~recursive(){}

void recursive::forward(tensor& output, const tensor& input){
    input.mult(output, weights);
    
    prev.mult(forwardTemp, weights);

    output.add(output, forwardTemp);
    output.add(output, bias);
}

void recursive::backward(tensor& dInput, const tensor& dOutput){}


std::string recursive::getLayerType(){return "Recersive";}
