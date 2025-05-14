#include "class_base.h"

BaseLearner::~BaseLearner(){}

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

void BaseLearner::checkpoint(std::ofstream& f){
    throw std::runtime_error("cannot use base learner class");
}

std::string BaseLearner::getLearnerType(){
    return "Base";
}



BaseLearnerSelector::~BaseLearnerSelector(){}

BaseLearner* BaseLearnerSelector::construct(tensor* t){
    throw std::runtime_error("cannot construct base learner selector");
}

BaseLearner* BaseLearnerSelector::load(tensor* t, std::ifstream& f){
    int n;
    f.read(reinterpret_cast<char*>(&n), sizeof(int));

    std::string s;
    s.resize(n);
    f.read(&s[0], sizeof(char)*n);

    if(s == AlphaLearner::getStaticLearnerType())
        return new AlphaLearner(t, f);
    if(s == MomentumLearner::getStaticLearnerType())
        return new MomentumLearner(t, f);
    if(s == AdagradLearner::getStaticLearnerType())
        return new AdagradLearner(t, f);
    if(s == AdamLearner::getStaticLearnerType())
        return new AdamLearner(t, f);
    
    throw std::runtime_error("learner not implemented in layer loader");
}

#include "vanilla/class_learner.cpp"
#include "adagrad/class_adagrad.cpp"
#include "momentum/class_momentum.cpp"
#include "adam/class_adam.cpp"
