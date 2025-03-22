#ifndef CLASS_LAYER_RECURSIVE_H
#define CLASS_LAYER_RECURSIVE_H

#include "../class_layer.h"

class recursive : public BaseLayer{
    private:
    tensor weights;
    tensor bias;
    tensor rWeights;
    tensor prev;
    tensor forwardTemp;

    public:
    virtual ~recursive();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&);

    std::string getLayerType();
};

#endif