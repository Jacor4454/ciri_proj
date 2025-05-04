#include "gradients.h"

#define grainputs float* output, float* data, float* correct, float* ign1, long n, long ign2, long ign3, long ign4, int offset, int step

void squaredError(grainputs){
    // e = x_bar^2
    for(int i = offset; i < n; i+=step)
        output[offset] += pow((correct[i] - data[i]), 2);
    
}
void squaredDiff(grainputs){
    // d = -2*x_bar
    for(int i = offset; i < n; i+=step)
        output[i] = -2*(correct[i] - data[i]);
}

void meanSquaredError(grainputs){
    // e = (x_bar^2)/N
    for(int i = offset; i < n; i+=step)
        output[offset] += pow((correct[i] - data[i]), 2);
    output[offset] /= n;
}
void meanSquaredDiff(grainputs){
    // d = (-2*x_bar)/N
    for(int i = offset; i < n; i+=step)
        output[i] = -2*(correct[i] - data[i]) / n;
}

void crossEntropyError(grainputs){
    // e = -(c*ln(x) + (1-c)*ln(1-x))
    for(int i = offset; i < n; i+=step)
        if(data[i] == 0){
            output[offset] += -1*((correct[i]*log(0.0000001))+((1-correct[i])*log(0.9999999)));
        } else if(data[i] == 1){
            output[offset] += -1*((correct[i]*log(0.9999999))+((1-correct[i])*log(0.0000001)));
        } else {
            output[offset] += -1*((correct[i]*log(data[i]))+((1-correct[i])*log(1-data[i])));
        }
}
void crossEntropyDiff(grainputs){
    // d = -((c/x) + (1-c)/(1-x))
    for(int i = offset; i < n; i+=step)
        if(data[i] == 0){
            output[i] = -1*((correct[i]/0.0000001)-((1-correct[i])/(0.9999999)));
        } else if(data[i] == 1){
            output[i] = -1*((correct[i]/0.9999999)-((1-correct[i])/(0.0000001)));
        } else {
            output[i] = -1*((correct[i]/data[i])-((1-correct[i])/(1-data[i])));
        }
}
