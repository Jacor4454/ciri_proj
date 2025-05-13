#include "class_learner.h"

AlphaLearner::AlphaLearner(tensor* orr, float alpha_):
    differ(*orr)
{
    original = orr;
    alpha = alpha_;
    differ.set(0.0);
}

AlphaLearner::~AlphaLearner(){}

void AlphaLearner::backprop(const tensor& a, const tensor& b){
    a.fastDeMultLInc(differ, b);
}

void AlphaLearner::backprop(const tensor& a){
    differ.add(differ, a);
}

void AlphaLearner::learn(){
    differ.alphaSub(*original, alpha);
}

void AlphaLearner::clear(){
    differ.set(0.0);
}

void AlphaLearner::checkpoint(std::ofstream& f){
    // write learner type
    std::string name = getLearnerType();
    int i_cache = name.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    f.write(name.c_str(), sizeof(char)*i_cache);

    // write data
    f.write(reinterpret_cast<const char*>(&alpha), sizeof(float));
}

std::string AlphaLearner::getLearnerType(){
    return "Vanilla";
}


AlphaLearnerSelector::AlphaLearnerSelector(float alpha_):alpha(alpha_){}
BaseLearner* AlphaLearnerSelector::construct(tensor* t){return new AlphaLearner(t, alpha);}
