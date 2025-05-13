#ifndef CLASS_ADAGRAD_LEARNER_H
#define CLASS_ADAGRAD_LEARNER_H

#include "../class_base.h"

class AdagradLearner : public BaseLearner{
    private:
    float alpha;
    tensor* original;
    tensor meanSquared;
    tensor differ;

    public:
    AdagradLearner(tensor*, float);// takes the tensor the user wants to track

    void backprop(const tensor&, const tensor&);
    void backprop(const tensor&);

    void learn();
    void clear();
    
    void checkpoint(std::ofstream& f);
    
    std::string getLearnerType();
};

class AdagradLearnerSelector : public BaseLearnerSelector{
    private:
    float alpha;

    public:
    AdagradLearnerSelector(float);
    BaseLearner* construct(tensor*);
};

#endif