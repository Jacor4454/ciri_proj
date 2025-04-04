#include "class_learner.h"

BaseLearner::BaseLearner(tensor* orr, float alpha_):
    differ(*orr)
{
    original = orr;
    alpha = alpha_;
}

void BaseLearner::backprop(const tensor& a, const tensor& b){
    a.fastDeMultLInc(differ, b);
}

void BaseLearner::backprop(const tensor& a){
    differ.add(differ, a);
}

void BaseLearner::learn(){
    differ.alphaSub(*original, alpha);
}

void BaseLearner::clear(){
    differ.sMult(differ, 0.0);// for now
}

