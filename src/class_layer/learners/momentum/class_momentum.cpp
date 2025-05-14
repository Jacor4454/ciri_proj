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

MomentumLearner::MomentumLearner(tensor* orr, std::ifstream& f):
differ(*orr),
meanSquared(f)
{
    alphas = (float*)malloc(sizeof(float)*2);
    original = orr;
    f.read(reinterpret_cast<char*>(alphas), sizeof(int)*2);
    differ.set(0.0);
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

void MomentumLearner::checkpoint(std::ofstream& f){
    // write learner type
    std::string name = getLearnerType();
    int i_cache = name.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    f.write(name.c_str(), sizeof(char)*i_cache);

    // write tensors
    meanSquared.save(f);

    // write data
    f.write(reinterpret_cast<const char*>(alphas), sizeof(float)*2);
}

std::string MomentumLearner::getLearnerType(){
    return getStaticLearnerType();
}
std::string MomentumLearner::getStaticLearnerType(){
    return "Momentum";
}

MomentumLearnerSelector::MomentumLearnerSelector(float alpha_, float beta_):alpha(alpha_),beta(beta_){}
BaseLearner* MomentumLearnerSelector::construct(tensor* t){return new MomentumLearner(t, alpha, beta);}

