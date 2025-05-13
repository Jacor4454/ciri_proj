#include "class_adagrad.h"

AdagradLearner::AdagradLearner(tensor* orr, float alpha_):
    differ(*orr),
    meanSquared(*orr)
{
    original = orr;
    alpha = alpha_;
    differ.set(0.0);
    meanSquared.set(0.0); // only set once
}

void AdagradLearner::backprop(const tensor& a, const tensor& b){
    a.fastDeMultLInc(differ, b);
}

void AdagradLearner::backprop(const tensor& a){
    differ.add(differ, a);
}

void AdagradLearner::learn(){
    meanSquared.adagrad(*original, differ, alpha);
}

void AdagradLearner::clear(){
    differ.set(0.0);
}

void AdagradLearner::checkpoint(std::ofstream& f){
    // write learner type
    std::string name = getLearnerType();
    int i_cache = name.size();
    f.write(reinterpret_cast<const char*>(&i_cache), sizeof(int));
    f.write(name.c_str(), sizeof(char)*i_cache);

    // write data
    f.write(reinterpret_cast<const char*>(&alpha), sizeof(float));

    // write tensors
    meanSquared.save(f);
}

std::string AdagradLearner::getLearnerType(){
    return "Adagrad";
}



AdagradLearnerSelector::AdagradLearnerSelector(float alpha_):alpha(alpha_){}
BaseLearner* AdagradLearnerSelector::construct(tensor* t){return new AdagradLearner(t, alpha);}
