#include "class_momentum.h"

MomentumLearner::MomentumLearner(tensor* orr, float alpha_, float beta_):
    differ(*orr),
    meanSquared(*orr)
{
    alphas = (float*)malloc(sizeof(float)*2);
    original = orr;
    alphas[0] = alpha_;
    alphas[1] = beta_;
    differ.set(0.0);
    meanSquared.set(0.0); // only set once
}

MomentumLearner::~MomentumLearner(){
    free(alphas);
}

void MomentumLearner::backprop(const tensor& a, const tensor& b){
    a.fastDeMultLInc(differ, b);
}

void MomentumLearner::backprop(const tensor& a){
    differ.add(differ, a);
}

void MomentumLearner::learn(){
    meanSquared.momentum(*original, differ, alphas);
}

void MomentumLearner::clear(){
    differ.set(0.0);
}


MomentumLearnerSelector::MomentumLearnerSelector(float alpha_, float beta_):alpha(alpha_),beta(beta_){}
BaseLearner* MomentumLearnerSelector::construct(tensor* t){return new MomentumLearner(t, alpha, beta);}

