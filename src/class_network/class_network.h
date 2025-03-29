#ifndef CLASS_NETWORK_H
#define CLASS_NETWORK_H

#include "../class_layer/class_layer.h"
#include "../class_layer/perceptron/class_perceptron.h"
#include "../class_layer/recersive/class_recersive.h"

namespace layers {
    typedef enum {
        perceptron,
        recursive,
    } layerTypes;
}

class inputDefObject {
    private:
    std::vector<int> inputDims;
    public:
    inputDefObject(std::vector<int> i):inputDims(i){}
    std::vector<int> getDims(){return inputDims;}
};
class layerDefObject {
    private:
    std::vector<int> outputDims;
    layers::layerTypes lyrTyp;
    activations::accTypes accTyp;
    public:
    layerDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
};
class outputDefObject {
    private:
    std::vector<int> outputDims;
    layers::layerTypes lyrTyp;
    activations::accTypes accTyp;
    public:
    outputDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
};


class network{
    protected:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    std::vector<tensor> ts;
    int N = 0;
    public:
    network(inputDefObject, std::vector<layerDefObject>, outputDefObject);
    ~network();

    void forward(tensor& input);
};


#endif