#include "class_learner.h"

AlphaLearner::AlphaLearner(tensor* orr, float alpha_):
    differ(*orr)
{
    original = orr;
    alpha = alpha_;
    differ.set(0.0);
}

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


AlphaLearnerSelector::AlphaLearnerSelector(float alpha_):alpha(alpha_){}
BaseLearner* AlphaLearnerSelector::construct(tensor* t){return new AlphaLearner(t, alpha);}
