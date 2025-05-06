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
    tensor hold;
    tensor dInt;
    tensor prev;
    activations::accTypes acc;
    BaseLearnerSelector* bls;
    

    public:
    recursive(std::vector<int>, std::vector<int>);
    void setAcc(activations::accTypes);
    void setLearners(BaseLearnerSelector*);
    virtual ~recursive();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&,  const tensor&);
    void learn();
    void clear();

    std::string getLayerType();
};

#endif