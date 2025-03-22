#ifndef CLASS_ABS_LAYER_H
#define CLASS_ABS_LAYER_H

#include "../class_tensor/class_tensor.h"

class BaseLayer{
    protected:
    std::vector<int> dims;

    public:
    virtual ~BaseLayer();

    virtual void forward(tensor&, const tensor&);
    virtual void backward(tensor&, const tensor&, const tensor&, const tensor&);
    std::vector<int> getDims();

    virtual std::string getLayerType();

};

#endif