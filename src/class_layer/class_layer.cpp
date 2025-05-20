#include "class_layer.h"

float inverse_sqrt(float number_){
    return 1/sqrt(number_);
}

std::default_random_engine BaseLayer::generator = std::default_random_engine();

// I/O to dims vector converter for the constructors
std::vector<int> BaseLayer::makeWeightsDims(std::vector<int> in, std::vector<int> out){
    std::vector<int> output(in);
    int x = output.size();
    output[x-2] = output[x-1];
    output[x-1] = out[x-1];
    return output;
}

// save as above but for the hidden weights
std::vector<int> BaseLayer::makeHWeightsDims(std::vector<int> out){
    std::vector<int> output(out);
    int x = output.size();
    output[x-2] = output[x-1];
    return output;
}

std::vector<int> BaseLayer::loadWeightsDims(std::ifstream& f){
    int n;

    f.read(reinterpret_cast<char*>(&n), sizeof(int));
    std::vector<int> output(n);
    f.read(reinterpret_cast<char*>(&output[0]), sizeof(int)*n);

    return output;
}

// load of errors to prevent implementing this class
void BaseLayer::setAcc(activations::accTypes a){throw std::runtime_error("cannot set acc type in base layer");}
void BaseLayer::setLearners(BaseLearnerSelector* bls){throw std::runtime_error("cannot set learner type in base layer");}
void BaseLayer::randomise(){throw std::runtime_error("cannot randomise learner type in base layer");}
BaseLayer::~BaseLayer(){}

void BaseLayer::forward(tensor& output, const tensor& input){throw std::runtime_error("cannot forward base layer");}
void BaseLayer::backward(tensor& dInput, const tensor& input, const tensor& dOutput, const tensor& output, const tensor&){throw std::runtime_error("cannot backward base layer");}
void BaseLayer::learn(){throw std::runtime_error("cannot learn base layer");}
void BaseLayer::clear(){throw std::runtime_error("cannot clear base layer");}

void BaseLayer::save(std::ofstream&){throw std::runtime_error("cannot save base layer");}
void BaseLayer::save_checkpoint(std::ofstream&){throw std::runtime_error("cannot save base layer");}
void BaseLayer::load_checkpoint(std::ifstream&){throw std::runtime_error("cannot load base layer");}

// return layer type name
std::string BaseLayer::getLayerType(){return "Base";}



#include "./perceptron/class_perceptron.cpp"
#include "./recersive/class_recersive.cpp"
