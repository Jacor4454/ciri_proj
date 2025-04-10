#include "activators.h"



void ReLU(float* data, long N){
    for(int i = 0; i < N; i++){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}
void deReLU(float* output, float* data, long N){
    for(int i = 0; i < N; i++){
        output[i] = data[i] > 0 ? 1 : 0;
    }
}
void Sigmoid(float* data, long N){
    for(int i = 0; i < N; i++){
        data[i] = (1/(1+(exp(-1*data[i]))));
    }
}
void deSigmoid(float* output, float* data, long N){
    for(int i = 0; i < N; i++){
        output[i] = data[i] * (1-data[i]);
    }
}
void Tanh(float* data, long N){
    for(int i = 0; i < N; i++){
        data[i] = tanh(data[i]);
    }
}
void deTanh(float* output, float* data, long N){
    for(int i = 0; i < N; i++){
        output[i] = 1-(data[i]*data[i]);
    }
}
void Softmax(float* data, long N){
    float total = 0;

    //totalise
    for(int i = 0; i < N; i++){
        total += exp(data[i]);//could create prepper but who has the time (future me thats who)
    }

    for(int i = 0; i < N; i++){
        data[i] = exp(data[i])/total;
    }
}
void deSoftmax(float* output, float* data, long N){
    for(int i = 0; i < N; i++){
        output[i] = data[i]*(1-data[i]);
    }
}

#define leakyness 0.01
void Leaky_relu(float* data, long N){
    for(int i = 0; i < N; i++){
        data[i] = (data[i] > 0) ? data[i] : data[i]*leakyness;
    }
}
void deLeaky_relu(float* output, float* data, long N){
    for(int i = 0; i < N; i++){
        output[i] = (data[i] > 0) ? 1.0 : leakyness;
    }
}

void Activate(float* data, long N, activations::accTypes a){
    switch(a){
        case activations::ReLU:
            ReLU(data, N);
            break;
        case activations::Sigmoid:
            Sigmoid(data, N);
            break;
        case activations::tanh:
            Tanh(data, N);
            break;
        case activations::softmax:
            Softmax(data, N);
            break;
        case activations::leakyReLU:
            Leaky_relu(data, N);
            break;
        default:
            throw std::runtime_error("activation type not supported yet");
    }
}

void DeActivate(float* data, float* output, long N, activations::accTypes a){
    switch(a){
        case activations::ReLU:
            deReLU(output, data, N);
            break;
        case activations::Sigmoid:
            deSigmoid(output, data, N);
            break;
        case activations::tanh:
            deTanh(output, data, N);
            break;
        case activations::softmax:
            deSoftmax(output, data, N);
            break;
        case activations::leakyReLU:
            deLeaky_relu(output, data, N);
            break;
        default:
            throw std::runtime_error("deactivation type not supported yet");
    }
}
