#ifndef CLASS_LAYER_RECURSIVE_H
#define CLASS_LAYER_RECURSIVE_H

#include "../class_layer.h"

class recursive : public BaseLayer{
    private:
    std::vector<int> weightsDims;
    std::vector<int> hweightsDims;
    tensor weights;
    tensor bias;
    tensor rWeights;
    BaseLearner* dweight;
    BaseLearner* drweight;
    BaseLearner* dbias;
    tensor dInt;
    tensor prev;
    

    public:
    recursive(std::vector<int>, std::vector<int>);
    virtual ~recursive();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    void learn();

    std::string getLayerType();
};

#endif