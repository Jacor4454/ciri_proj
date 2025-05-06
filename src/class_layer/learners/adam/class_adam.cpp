#include "class_adam.h"

AdamLearner::AdamLearner(tensor* orr, float alpha_, float beta1_, float beta2_):
    differ(*orr),
    momentum(*orr),
    velocity(*orr)
{
    alphas = (float*)malloc(sizeof(float)*3);
    alphas[0] = alpha_;
    alphas[1] = beta1_;
    alphas[2] = beta2_;
    alphas[3] = 1;
    alphas[4] = 1;
    original = orr;
    differ.set(0.0);
    momentum.set(0.0); // only set once
    velocity.set(0.0); // only set once
}

AdamLearner::~AdamLearner(){
    free(alphas);
}

void AdamLearner::backprop(const tensor& a, const tensor& b){
    a.fastDeMultLInc(differ, b);
}

void AdamLearner::backprop(const tensor& a){
    differ.add(differ, a);
}

void AdamLearner::learn(){
    momentum.adam_m(differ, alphas);
    velocity.adam_v(differ, alphas);
    velocity.adam_c(*original, momentum, alphas);
}

void AdamLearner::clear(){
    differ.set(0.0);
}


AdamLearnerSelector::AdamLearnerSelector(float alpha_, float beta1_, float beta2_):alpha(alpha_),beta1(beta1_),beta2(beta2_){}
BaseLearner* AdamLearnerSelector::construct(tensor* t){return new AdamLearner(t, alpha, beta1, beta2);}

