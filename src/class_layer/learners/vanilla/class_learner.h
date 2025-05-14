#ifndef CLASS_VANILLA_LEARNER_H
#define CLASS_VANILLA_LEARNER_H

#include "../class_base.h"

class AlphaLearner : public BaseLearner{
    private:
    float alpha;
    tensor* original;

    public:
    tensor differ;
    AlphaLearner(tensor*, float);// takes the tensor the user wants to track
    AlphaLearner(tensor*, std::ifstream&);
    ~AlphaLearner();

    void backprop(const tensor&, const tensor&);
    void backprop(const tensor&);

    void learn();
    void clear();
    
    void checkpoint(std::ofstream& f);
    
    std::string getLearnerType();
    static std::string getStaticLearnerType();
};

class AlphaLearnerSelector : public BaseLearnerSelector{
    private:
    float alpha;

    public:
    AlphaLearnerSelector(float);
    BaseLearner* construct(tensor*);
};

#endif