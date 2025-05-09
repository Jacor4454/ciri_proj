#include "class_tensor.h"

#define inputs float* output, float* a, float* b, float* c, long n, long m, long k, long block, int offset, int step

float invSqrt(float f){
    return 1/sqrt(f);
}

// shadies and funcs
// adds 2 matrix's into output
void add_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] + b[i];
}

// placewise multiply 2 matrix's into output
void s_mult_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] * b[i];
}

// adds a constant into output
void add_K_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] + (*b);
}

// multiplies by a constant into output
void s_mult_K_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step)
        output[i] = a[i] * (*b);
}

// matrix multiply, splitting job by M
void mult_M_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot;
            }
        }
    }
}
// matrix multiply, splitting job by N
void mult_N_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot;
            }
        }
    }
}

// matrix multiply, inverting the left(a) matrix, splitting job by M
void deMultL_M_skip_shadie(inputs){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}
// matrix multiply, inverting the left(a) matrix, splitting job by N
void deMultL_N_skip_shadie(inputs){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}

// matrix multiply, inverting the left(a) matrix, cumulative output, splitting job by M
void deMultLInc_M_skip_shadie(inputs){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] += tot;
            }
        }
    }
}
// matrix multiply, inverting the left(a) matrix, cumulative output, splitting job by N
void deMultLInc_N_skip_shadie(inputs){
    // y == t.y
    // output.x = t.x
    // output.y = x

    // a = kxn tb = kxm

    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*k + k_)*n + i] * b[(blk*k + k_)*m + j];
                }
                output[(blk*n + i)*m + j] += tot;
            }
        }
    }
}

// matrix multiply, inverting the right(b) matrix, splitting job by M
void deMultR_M_skip_shadie(inputs){
    // x == t.x
    // output.x = t.y
    // output.y = y

    // a = nxk tb = mxk

    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*n + i)*k + k_] * b[(blk*m + j)*k + k_];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}
// matrix multiply, inverting the right(b) matrix, splitting job by N
void deMultR_N_skip_shadie(inputs){
    // x == t.x
    // output.x = t.y
    // output.y = y

    // a = nxk tb = mxk

    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[(blk*n + i)*k + k_] * b[(blk*m + j)*k + k_];
                }
                output[(blk*n + i)*m + j] = tot;
            }
        }
    }
}

// multiply a and b, add c and output, splitting job by M
void multNadd_M_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot + c[((blk*n + i)*m) + j];
            }
        }
    }
}
// multiply a and b, add c and output, splitting job by M
void multNadd_N_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] = tot + c[((blk*n + i)*m) + j];
            }
        }
    }    
}

// multiply a and b, increment output by result, splitting job by M
void multNInc_M_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = 0; i < n; i++){
            for(int j = offset; j < m; j+=step){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] += tot;
            }
        }
    }
}
// multiply a and b, increment output by result, splitting job by N
void multNInc_N_skip_shadie(inputs){
    // checks are performed off thread
    for(int blk = 0; blk < block; blk++){
        for(int i = offset; i < n; i+=step){
            for(int j = 0; j < m; j++){
                float tot = 0;
                for(int k_ = 0; k_ < k; k_++){
                    tot += a[((blk*n + i)*k) + k_] * b[((blk*k + k_)*m) + j];
                }
                output[((blk*n + i)*m) + j] += tot;
            }
        }
    }
}

// for learning
void alpha_sub(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        output[i] -= a[i] * *b;
    }
}

void adagrad_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        b[i] += a[i]*a[i];
        output[i] -= (*c) * a[i] * invSqrt(b[i]+0.00000001);
    }
}

// https://medium.com/@akshayush007/journey-llm-7-optimization-algorithms-wrt-deep-learning-04a8cec8fc25
// moving average: https://medium.com/analytics-vidhya/momentum-a-simple-yet-efficient-optimizing-technique-ef76834e4423
// idk what this is
// void momentum_shadie(inputs){
//     // checks are performed off thread
//     for(int i = offset; i < n; i+=step){
//         b[i] = (b[i]*c[1]) + (a[i]*(1-c[1]));
//         output[i] -= b[i] * c[0];
//     }
// }

void momentum_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        b[i] = (b[i]*c[1]) + (a[i]*c[0]);
        output[i] -= b[i];
    }
}


// https://www.geeksforgeeks.org/adam-optimizer/
//adams
void adam_m_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        a[i] = (a[i]*c[1]) + (1-c[1])*b[i];
    }
}
void adam_v_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        a[i] = (a[i]*c[2]) + (1-c[2])*b[i]*b[i];
    }
}
void adam_c_shadie(inputs){
    // checks are performed off thread
    for(int i = offset; i < n; i+=step){
        float mb = a[i]/(1-c[3]);
        float vb = b[i]/(1-c[4]);
        output[i] -= mb * invSqrt(vb+0.0000001) * c[0];
    }
}
