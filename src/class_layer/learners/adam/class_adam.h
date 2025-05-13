#ifndef CLASS_ADAM_LEARNER_H
#define CLASS_ADAM_LEARNER_H

#include "../class_base.h"

class AdamLearner : public BaseLearner{
    private:
    float* alphas;
    tensor* original;
    tensor momentum;
    tensor velocity;
    tensor differ;

    public:
    AdamLearner(tensor*, float, float, float);// takes the tensor the user wants to track
    ~AdamLearner();

    void backprop(const tensor&, const tensor&);
    void backprop(const tensor&);

    void learn();
    void clear();
    
    void checkpoint(std::ofstream& f);
    
    std::string getLearnerType();
};

class AdamLearnerSelector : public BaseLearnerSelector{
    private:
    float alpha;
    float beta1;
    float beta2;

    public:
    AdamLearnerSelector(float, float, float);
    BaseLearner* construct(tensor*);
};

#endif