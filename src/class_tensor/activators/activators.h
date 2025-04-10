#ifndef CLASS_ACTIVATOR_H
#define CLASS_ACTIVATOR_H

#include <math.h>
#include <stdexcept>

namespace activations {
    typedef enum{
        ReLU,
        Sigmoid,
        tanh,
        softmax,
        leakyReLU,
    } accTypes;
}

void Activate(float*, long, activations::accTypes);
void DeActivate(float*, float*, long, activations::accTypes);

#endif