#ifndef CLASS_LAYER_PERCEPTRON_H
#define CLASS_LAYER_PERCEPTRON_H

#include "../class_layer.h"

class perceptron : public BaseLayer{
    private:
    std::vector<int> weightsDims;
    tensor weights;
    tensor bias;
    BaseLearner* dweight;
    BaseLearner* dbias;
    tensor hold;

    public:
    perceptron(std::vector<int>, std::vector<int>);
    virtual ~perceptron();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    void learn();

    std::string getLayerType();
};

#endif