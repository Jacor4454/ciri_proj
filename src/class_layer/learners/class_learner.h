#ifndef CLASS_ABS_LEARNER_H
#define CLASS_ABS_LEARNER_H

#include "../class_layer.h"

class BaseLearner{
    protected:
    float alpha;
    tensor* original;

    public:
    tensor differ;
    BaseLearner(tensor*, float);// takes the tensor the user wants to track

    virtual void backprop(const tensor&, const tensor&);
    virtual void backprop(const tensor&);

    virtual void learn();
    virtual void clear();
};

#endif