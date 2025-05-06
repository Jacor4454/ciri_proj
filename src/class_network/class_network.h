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
    BaseLearnerSelector* oppTyp;
    public:
    layerDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_, BaseLearnerSelector* oppTyp_=nullptr):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_; oppTyp = oppTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
    BaseLearnerSelector* getOppTyp(){return oppTyp;}
};
class outputDefObject {
    private:
    std::vector<int> outputDims;
    layers::layerTypes lyrTyp;
    activations::accTypes accTyp;
    errors::errTypes errTyp;
    BaseLearnerSelector* oppTyp;
    public:
    outputDefObject(std::vector<int> o, layers::layerTypes lyrTyp_, activations::accTypes accTyp_, errors::errTypes errTyp_=errors::SE, BaseLearnerSelector* oppTyp_=nullptr):outputDims(o){lyrTyp = lyrTyp_; accTyp = accTyp_; errTyp = errTyp_; oppTyp = oppTyp_;}
    std::vector<int> getDims(){return outputDims;}
    layers::layerTypes getLyrTyp(){return lyrTyp;}
    activations::accTypes getAccTyp(){return accTyp;}
    errors::errTypes getErrTyp(){return errTyp;}
    BaseLearnerSelector* getOppTyp(){return oppTyp;}
};


class network{
    private:
    std::vector<BaseLayer*> layers;
    std::vector<std::vector<int>> dimss;
    std::vector<tensor> ts;
    int N = 0;
    public:
    static BaseLayer* getLayer(layers::layerTypes, std::vector<int>&, std::vector<int>&);
    network(inputDefObject, std::vector<layerDefObject>, outputDefObject);
    ~network();

    void forward(const std::vector<tensor>& input);
};


#endif