#ifndef CLASS_BASE_LEARNER_H
#define CLASS_BASE_LEARNER_H

#include "../../class_tensor/class_tensor.h"

class BaseLearner{
    public:
    virtual void backprop(const tensor&, const tensor&);
    virtual void backprop(const tensor&);

    virtual void learn();
    virtual void clear();
};

class BaseLearnerSelector{
    public:
    virtual BaseLearner* construct(tensor*);
};

#include "vanilla/class_learner.h"
#include "adagrad/class_adagrad.h"
#include "momentum/class_momentum.h"
#include "adam/class_adam.h"

#endif