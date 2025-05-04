#include "activators.h"

#define accinputs float* output, float* data, float* ign5, float* ign1, long n, long ign2, long ign3, long ign4, int offset, int step


void ReLU(accinputs){
    for(int i = offset; i < n; i+=step){
        data[i] = data[i] < 0 ? 0 : data[i];
    }
}
void deReLU(accinputs){
    for(int i = offset; i < n; i+=step){
        output[i] = data[i] > 0 ? 1 : 0;
    }
}
void Sigmoid(accinputs){
    for(int i = offset; i < n; i+=step){
        data[i] = (1/(1+(exp(-1*data[i]))));
    }
}
void deSigmoid(accinputs){
    for(int i = offset; i < n; i+=step){
        output[i] = data[i] * (1-data[i]);
    }
}
void Tanh(accinputs){
    for(int i = offset; i < n; i+=step){
        data[i] = tanh(data[i]);
    }
}
void deTanh(accinputs){
    for(int i = offset; i < n; i+=step){
        output[i] = 1-(data[i]*data[i]);
    }
}
// void Softmax(float* data, long N){
//     float total = 0;

//     //totalise
//     for(int i = 0; i < N; i++){
//         total += exp(data[i]);//could create prepper but who has the time (future me thats who)
//     }

//     for(int i = 0; i < N; i++){
//         data[i] = exp(data[i])/total;
//     }
// }
// void deSoftmax(float* output, float* data, long N){
//     for(int i = 0; i < N; i++){
//         output[i] = data[i]*(1-data[i]);
//     }
// }

#define ReLU_leakyness 0.01
void Leaky_relu(accinputs){
    for(int i = offset; i < n; i+=step){
        data[i] = (data[i] > 0) ? data[i] : data[i]*ReLU_leakyness;
    }
}
void deLeaky_relu(accinputs){
    for(int i = offset; i < n; i+=step){
        output[i] = (data[i] > 0) ? 1.0 : ReLU_leakyness;
    }
}
