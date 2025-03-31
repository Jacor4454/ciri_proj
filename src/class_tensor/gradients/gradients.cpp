#include "gradients.h"

float squaredError(float* data, float* correct, long N){
    float output = 0;
    for(int i = 0; i < N; i++)
        output += pow((correct[i] - data[i]), 2);
    return output;
}
void squaredDiff(float* output, float* data, float* correct, long N){
    for(int i = 0; i < N; i++)
        output[i] = -2*(correct[i] - data[i]);
}

float meanSquaredError(float* data, float* correct, long N){
    float output = 0;
    for(int i = 0; i < N; i++)
        output += pow((correct[i] - data[i]), 2)/(N);
    return output;
}
void meanSquaredDiff(float* output, float* data, float* correct, long N){
    for(int i = 0; i < N; i++)
        output[i] = -2*(correct[i] - data[i]) / N;
}

float crossEntropyError(float* data, float* correct, long N){
    float output = 0;
    for(int i = 0; i < N; i++)
        if(data[i] == 0){
            output += -1*((correct[i]*log(0.0000001))+((1-correct[i])*log(0.9999999)));
        } else if(data[i] == 1){
            output += -1*((correct[i]*log(0.9999999))+((1-correct[i])*log(0.0000001)));
        } else {
            output += -1*((correct[i]*log(data[i]))+((1-correct[i])*log(1-data[i])));
        }
    return output;
}
void crossEntropyDiff(float* output, float* data, float* correct, long N){
    for(int i = 0; i < N; i++)
        if(data[i] == 0){
            output[i] = -1*((correct[i]/0.0000001)-((1-correct[i])/(0.9999999)));
        } else if(data[i] == 1){
            output[i] = -1*((correct[i]/0.9999999)-((1-correct[i])/(0.0000001)));
        } else {
            output[i] = -1*((correct[i]/data[i])-((1-correct[i])/(1-data[i])));
        }
}

float Loss(float* data, float* correct, long N, errors::errTypes e){
    float output = 0;
    switch(e){
        case errors::SE:
            output = squaredError(data, correct, N);
            break;
        case errors::MSE:
            output = meanSquaredError(data, correct, N);
            break;
        case errors::CE:
            output = crossEntropyError(data, correct, N);
            break;
        default:
            throw std::runtime_error("loss type not supported yet");
    }
    return output;
}

void Gradient(float* output, float* data, float* correct, long N, errors::errTypes e){
    switch(e){
        case errors::SE:
            squaredDiff(output, data, correct, N);
            break;
        case errors::MSE:
            meanSquaredDiff(output, data, correct, N);
            break;
        case errors::CE:
            crossEntropyDiff(output, data, correct, N);
            break;
        default:
            throw std::runtime_error("loss type not supported yet");
    }
}
