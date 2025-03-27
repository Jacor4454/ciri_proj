#ifndef CLASS_ABS_LAYER_H
#define CLASS_ABS_LAYER_H

#include "../class_tensor/class_tensor.h"

class BaseLayer{

    public:
    static std::vector<int> makeWeightsDims(std::vector<int> in, std::vector<int> out);
    static std::vector<int> makeHWeightsDims(std::vector<int> out);

    virtual ~BaseLayer();

    virtual void forward(tensor&, const tensor&);
    virtual void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    virtual void learn(float alpha);
    
    virtual std::string getLayerType();

};

#endif