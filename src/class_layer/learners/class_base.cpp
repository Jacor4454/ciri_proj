#include "class_base.h"

void BaseLearner::backprop(const tensor& a, const tensor& b){
    throw std::runtime_error("cannot use base learner class");
}

void BaseLearner::backprop(const tensor& a){
    throw std::runtime_error("cannot use base learner class");
}

void BaseLearner::learn(){
    throw std::runtime_error("cannot use base learner class");
}

void BaseLearner::clear(){
    throw std::runtime_error("cannot use base learner class");
}

BaseLearner* BaseLearnerSelector::construct(tensor* t){
    throw std::runtime_error("cannot construct base learner selector");
}

#include "vanilla/class_learner.cpp"
#include "adagrad/class_adagrad.cpp"
#include "momentum/class_momentum.cpp"
#include "adam/class_adam.cpp"
