#include "class_adam.h"

AdamLearner::AdamLearner(tensor* orr, float alpha_, float beta1_, float beta2_):
    differ(*orr),
    momentum(*orr),
    velocity(*orr)
{
    alphas = (float*)malloc(sizeof(float)*5);
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
    alphas[3] *= alphas[1];
    alphas[4] *= alphas[2];
    velocity.adam_c(*original, momentum, alphas);
}

void AdamLearner::clear(){
    differ.set(0.0);
}

void AdamLearner::checkpoint(std::ofstream& f){
    // write learner type
    std::string name = getLearnerType();
    int i_cache = name.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    f.write(name.c_str(), sizeof(char)*i_cache);

    // write data
    f.write(reinterpret_cast<const char*>(alphas), sizeof(float)*5);

    // write tensors
    momentum.save(f);
    velocity.save(f);
}

std::string AdamLearner::getLearnerType(){
    return "Adam";
}



AdamLearnerSelector::AdamLearnerSelector(float alpha_, float beta1_, float beta2_):alpha(alpha_),beta1(beta1_),beta2(beta2_){}
BaseLearner* AdamLearnerSelector::construct(tensor* t){return new AdamLearner(t, alpha, beta1, beta2);}

