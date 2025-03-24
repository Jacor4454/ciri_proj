#ifndef CLASS_LAYER_PERCEPTRON_H
#define CLASS_LAYER_PERCEPTRON_H

#include "../class_layer.h"

class perceptron : public BaseLayer{
    private:
    tensor weights;
    tensor bias;
    tensor dweights;
    tensor dbias;
    tensor dweightsTemp;
    tensor hold;

    public:
    perceptron(int in, int out);
    virtual ~perceptron();

    void forward(tensor&, const tensor&);
    void backward(tensor&, const tensor&, const tensor&, const tensor&, tensor&);
    void learn(float alpha);

    std::string getLayerType();
};

#endif