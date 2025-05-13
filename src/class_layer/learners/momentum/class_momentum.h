#ifndef CLASS_MOMENTUM_LEARNER_H
#define CLASS_MOMENTUM_LEARNER_H

#include "../class_base.h"

class MomentumLearner : public BaseLearner{
    private:
    float* alphas;
    tensor* original;
    tensor meanSquared;
    tensor differ;

    public:
    MomentumLearner(tensor*, float, float);// takes the tensor the user wants to track
    ~MomentumLearner();

    void backprop(const tensor&, const tensor&);
    void backprop(const tensor&);

    void learn();
    void clear();
    
    void checkpoint(std::ofstream& f);
    
    std::string getLearnerType();
};

class MomentumLearnerSelector : public BaseLearnerSelector{
    private:
    float alpha;
    float beta;

    public:
    MomentumLearnerSelector(float, float);
    BaseLearner* construct(tensor*);
};

#endif