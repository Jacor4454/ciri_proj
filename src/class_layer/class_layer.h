#ifndef CLASS_ABS_LAYER_H
#define CLASS_ABS_LAYER_H

#include "../class_tensor/class_tensor.h"
#include "./learners/class_learner.h"

float inverse_sqrt(float number_);

class BaseLayer{

    public:
    static std::vector<int> makeWeightsDims(std::vector<int> in, std::vector<int> out);
    static std::vector<int> makeHWeightsDims(std::vector<int> out);
    static std::default_random_engine generator;

    virtual void setAcc(activations::accTypes);
    virtual ~BaseLayer();

    virtual void forward(tensor&, const tensor&);
    virtual void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    virtual void learn();
    virtual void clear();
    
    virtual std::string getLayerType();

};

#endif