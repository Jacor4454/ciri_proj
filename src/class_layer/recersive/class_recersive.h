#ifndef CLASS_LAYER_RECURSIVE_H
#define CLASS_LAYER_RECURSIVE_H

#include "../class_layer.h"

class recursive : public BaseLayer{
    private:
    tensor weights;
    tensor bias;
    tensor rWeights;
    tensor prev;
    tensor dInt;
    tensor dbias;
    tensor dweights;
    tensor dweightsTemp;
    tensor dRWeights;
    tensor dRWeightsTemp;

    public:
    recursive(int in, int out);
    virtual ~recursive();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    void learn(float alpha);

    std::string getLayerType();
};

#endif