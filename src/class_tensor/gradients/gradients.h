#ifndef CLASS_GRADIENTS_H
#define CLASS_GRADIENTS_H

#include <math.h>
#include <stdexcept>

namespace errors {
    typedef enum{
        SE, // Square Error
        MSE, // Mean Squared Error
        CE, // Cross-Entropy
    } errTypes;
}

float Loss(float*, float*, long, errors::errTypes);
void Gradient(float*, float*, float*, long, errors::errTypes);

#endif