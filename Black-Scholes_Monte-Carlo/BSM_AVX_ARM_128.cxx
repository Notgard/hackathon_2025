/*

   Monte Carlo Hackathon created by Hafsa Demnati and Patrick Demichel @ Viridien 2024
   The code compute a Call Option with a Monte Carlo method and compare the result with the analytical equation of Black-Scholes Merton : more details in the documentation

   Compilation : g++ -O BSM.cxx -o BSM

   Exemple of run: ./BSM #simulations #runs

   We want to measure 1000 runs and get the average error below a specific level
   Adjust the parameter #simulations to achieve the expected Average Relative Error

   points given for achieving Average Relative Error for 1000 runs < Average Relative Error: 0.01%     : short           ~20mn tuned and all cores
   points given for achieving Average Relative Error for 1000 runs < Average Relative Error: 0.005%    : normal          ~1h
   points given for achieving Average Relative Error for 1000 runs < Average Relative Error: 0.002%    : long            ~8h
   points given for achieving Average Relative Error for 1000 runs < Average Relative Error: 0.001%    : super long :    ~24h

   You can observe that from run to run there is a small difference caused using a different seed
   Deliver the full logs that show the randomly selected seed ; it will permit us to raproduce and verify the results

   You need to run 10 times the program; with the same parameter 1 #simulations and 1000 as parameter 2

   The performance is printed in the logs : more points given for each objective to the team with the best performance, the second, third and so on ...

   0.773595%    0.896091%      0.5748%    0.621321%    0.620323%    0.854219%    0.697301%    0.526567%    0.607043%    0.906975% ./BSM 100000    10
    0.75403%    0.727078%     0.63101%    0.753609%    0.733543%    0.728597%    0.753131%    0.859521%    0.696769%    0.699988% ./BSM 100000    100

   0.282992%    0.181664%    0.317491%    0.254558%    0.194851%     0.22103%   0.0953011%    0.250809%    0.310949%    0.211331% ./BSM 1000000   10
   0.224017%    0.230809%    0.239547%    0.217105%    0.258575%      0.1944%    0.228919%    0.258778%    0.235938%     0.25739% ./BSM 1000000   100

   0.056911%   0.0929754%   0.0599475%   0.0681029%   0.0618026%    0.128031%   0.0389641%   0.0588954%   0.0651689%    0.122257% ./BSM 10000000  10
  0.0625289%   0.0785358%   0.0781138%   0.0781734%   0.0736234%   0.0811247%    0.076021%   0.0773279%   0.0867399%   0.0765197% ./BSM 10000000  100

  0.0200822%   0.0257806%   0.0207118%   0.0179176%   0.0191748%    0.024724%   0.0185942%   0.0138896%    0.027215%   0.0257985% ./BSM 100000000 10
  0.0227214%   0.0213892%   0.0198618%   0.0229917%   0.0213438%   0.0252195%   0.0235354%    0.022934%   0.0243098%   0.0221371% ./BSM 100000000 100

   As you can see the first parameter define the average precision
   The second parameter as an average of multiple runs offer a smaller volativity of the result; that's why we ask for 1000 runs as second parameter "imposed"
   You can run smaller values of parameter 2 while you experiment ; but for the final results use strictly 1000

   The performance is somehow linear with the parameter 1 then multiple actions are expected to achieve all objectives
   Using the internet of chatgpt you can find and use another random generator; but you need to achieve similar numerical results since we use BSM algorithm to verify we are OK
   Except if you have a Nobel Price, you cannot change the code not measured by the performance mecanism
   You can use any method of parallelization or optimization
   You can use any compiler; vectorization; trigonometric library; we judge only numericla precision and performance

   Provide the traces of the 10 runs

*/

#include <iostream>
#include <cmath>
#include <random>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip> // For setting precision
#include <cstdint>
#include <cstddef>
#include <sys/time.h>
#include <armpl.h>
#include <bitset>
#include <climits>
#include <cstring>
#include <iostream>

#define SIMDE_ENABLE_NATIVE_ALIASES
#include "simde/simde/x86/avx2.h"
#include "simde/simde/x86/fma.h"

#define assert_message(cond, msg)      \
    if (!(cond))                       \
    {                                  \
        std::cerr << msg << std::endl; \
        abort();                       \
    }

#define ui64 u_int64_t
#ifndef REAL
    #define REAL double
#endif  
using real = REAL;

#define VECTOR_SIZE 4

// Generate a random seed at the start of the program using random_device
std::random_device rd;
unsigned long long global_seed = rd(); // This will be the global seed

double
dml_micros()
{
    static struct timezone tz;
    static struct timeval tv;
    gettimeofday(&tv, &tz);
    return ((tv.tv_sec * 1000000.0) + tv.tv_usec);
}

// Function to generate Gaussian noise using Box-Muller transform
real gaussian_box_muller()
{
    static std::mt19937 generator(global_seed);
    static std::normal_distribution<real> distribution(0.0, 1.0);
    return distribution(generator);

    /*return */
}
inline real real_sqrt(real r)
{
    if constexpr (sizeof(real) == sizeof(double)) return sqrt(r);
    if constexpr (sizeof(real) == sizeof(float)) return sqrtf(r);
    if constexpr (sizeof(real) == sizeof(long double)) return sqrtl(r);
}

void arm_random_number_generator(float * random_numbers, size_t size, VSLStreamStatePtr stream)
{
    int errcode = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, stream, size, random_numbers, 0.0, 1.0);
    if (errcode != VSL_STATUS_OK)
    {
       assert_message(errcode == VSL_ERROR_OK, "vsRngGaussian failed");
    }
}

// Helper function: Approximation for vectorized exp
__m128 exp128_ps(__m128 x) {
    // Approximation or use a library like Sleef for accurate vectorized exp
    // This is a placeholder; replace with a proper implementation.
    float x_array[VECTOR_SIZE];
    float result_array[VECTOR_SIZE];
    _mm128_storeu_ps(x_array, x);
    for (int i = 0; i < VECTOR_SIZE; ++i) {
        result_array[i] = std::exp(x_array[i]);
    }
    return _mm128_loadu_ps(result_array);
}

//function assumes all random numbers are generated beforehand
void monte_carlo_simulation_avx(
    float* random_numbers, 
    size_t num_simulations, 
    float S0, float temp_val, float T_sqrt, float K, 
    float& sum_payoffs, int offset)
{
    __m128 S0_vec = _mm128_set1_ps(S0);             // Broadcast S0 to all lanes
    __m128 temp_val_vec = _mm128_set1_ps(temp_val);   // Broadcast temp_val
    __m128 T_sqrt_vec = _mm128_set1_ps(T_sqrt);     // Broadcast T_sqrt
    __m128 K_vec = _mm128_set1_ps(K);               // Broadcast K
    __m128 zero_vec = _mm128_setzero_ps();          // Zero vector

    float thread_sum = 0.0f;                        // Local sum for reduction

    #pragma omp parallel for schedule(static, VECTOR_SIZE) reduction(+:thread_sum)
    for (size_t i = 0; i < num_simulations; i += VECTOR_SIZE) {
        // Load VECTOR_SIZE random numbers into a vector
        __m128 Z_vec = _mm128_loadu_ps(&random_numbers[i + offset]); // Unaligned load

        // Calculate exponent = temp_val + T_sqrt * Z
        __m128 exponent = _mm128_fmadd_ps(T_sqrt_vec, Z_vec, temp_val_vec);

        // Calculate ST = S0 * exp(exponent)
        __m128 exp_result = exp128_ps(exponent);     // Vectorized exp function
        __m128 ST_vec = _mm128_mul_ps(S0_vec, exp_result);

        // Calculate payoff = max(ST - K, 0.0)
        __m128 payoff_vec = _mm128_max_ps(_mm128_sub_ps(ST_vec, K_vec), zero_vec);

        // Accumulate payoffs into a local sum
        float sum_array[VECTOR_SIZE];
        _mm128_storeu_ps(sum_array, payoff_vec);
        for (int j = 0; j < VECTOR_SIZE; ++j) {
            thread_sum += sum_array[j];
        }
    }

    // Update the shared sum_payoffs variable
    sum_payoffs += thread_sum;
    //std::cout << "sum_payoffs= " << sum_payoffs << std::endl;
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs = std::stoull(argv[2]);

    // Input parameters
    ui64 S0 = 100;      // Initial stock price
    ui64 K = 110;       // Strike price
    real T = 1.0;     // Time to maturity (1 year)
    real r = 0.06;    // Risk-free interest rate
    real sigma = 0.2; // Volatility
    real q = 0.03;    // Dividend yield
    VSLStreamStatePtr stream;

    int errcode = vslNewStream(&stream, VSL_BRNG_MCG31, 1);
    assert_message(errcode == VSL_STATUS_OK, "vslNewStream failed");

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] << std::endl;

    real sum=0.0;
    float result = 0.0f;

    float *random_numbers = (float *)malloc(num_simulations * num_runs * sizeof(float));

    arm_random_number_generator(random_numbers, num_simulations * num_runs, stream);

    double t1=dml_micros();

    int offset = 0;

    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    for (ui64 run = 0; run < num_runs; ++run) {
        result = 0.0f;
        //sum+= black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations, stream);
        real temp_val = (r - q - 0.5 * sigma * sigma) * T;
        real T_sqrt = sigma * real_sqrt(T);
        monte_carlo_simulation_avx(random_numbers, num_simulations, S0, temp_val, T_sqrt, K, result, offset);
        
        result = (std::exp(-r * T) * result / num_simulations);
            
        offset += num_simulations;

        sum += result;
    }

    double t2=dml_micros();

    free(random_numbers);

    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;

    return 0;
}
