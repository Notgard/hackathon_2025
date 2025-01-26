/* 



    Monte Carlo Hackathon created by Hafsa Demnati and Patrick Demichel @ Viridien 2024

    The code compute a Call Option with a Monte Carlo method and compare the result with the analytical equation of Black-Scholes Merton : more details in the documentation



    Compilation : g++ -O BSM.cxx -o BSM



    Exemple of run: ./BSM #simulations #runs



./BSM 100 1000000

Global initial seed: 21852687      argv[1]= 100     argv[2]= 1000000

 value= 5.136359 in 10.191287 seconds



./BSM 100 1000000

Global initial seed: 4208275479      argv[1]= 100     argv[2]= 1000000

 value= 5.138515 in 10.223189 seconds

 

   We want the performance and value for largest # of simulations as it will define a more precise pricing

   If you run multiple runs you will see that the value fluctuate as expected

   The large number of runs will generate a more precise value then you will converge but it require a large computation



   give values for ./BSM 100000 1000000        

               for ./BSM 1000000 1000000

               for ./BSM 10000000 1000000

               for ./BSM 100000000 1000000

  

   We give points for best performance for each group of runs 



   You need to tune and parallelize the code to run for large # of simulations



*/

#include <iostream>
#include <cmath>
#include <random>
#include <cstdint>
#include <cstddef>
#include <sys/time.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <iomanip> // For setting precision
#include <xsimd/xsimd.hpp> // Include xsimd library for optimized mathematical operations
#include <omp.h>
#include "pcg_random.hpp"


#define ui64 u_int64_t
#ifndef REAL
    #define REAL double
#endif  
using real = REAL;

// Generate a random seed at the start of the program using random_device
std::random_device rd;
unsigned long long global_seed = rd(); // This will be the global seed

double dml_micros()
{
    static struct timezone tz;
    static struct timeval tv;
    gettimeofday(&tv, &tz);
    return ((tv.tv_sec * 1000000.0) + tv.tv_usec);
}

inline real real_sqrt(real r)
{
    if constexpr (sizeof(real) == sizeof(double)) return sqrt(r);
    if constexpr (sizeof(real) == sizeof(float)) return sqrtf(r);
    if constexpr (sizeof(real) == sizeof(long double)) return sqrtl(r);
}

// Function to calculate the Black-Scholes call option price using Monte Carlo method
real black_scholes_monte_carlo(ui64 S0, ui64 K, real T, real r, real sigma, real q, ui64 num_simulations, real expRT, real T_sqrt, real temp_val, real inv_num_sim) {
    using batch = xsimd::batch<real>; // Batch SIMD
    constexpr size_t simd_size = batch::size;

    // Creation of different batch needed
    const batch batch_temp_val(temp_val);
    const batch batch_T_sqrt(T_sqrt);
    const batch batch_S0(S0);
    const batch batch_K(K);
    const batch batch_zero(0.0);

    // random numbers generation with parallel area
    real* precomputed_random = new real[num_simulations];

    #pragma omp parallel
    {
        pcg32 generator; // Thread-safe random generator
        generator.seed(global_seed);
        std::normal_distribution<real> distribution(0.0, 1.0);

        #pragma omp for schedule(static)
        for (ui64 i = 0; i < num_simulations; ++i) {
            precomputed_random[i] = distribution(generator);  
        }
    }

    real sum_payoffs = 0.0;

    // Parallel area for computation of results
    #pragma omp parallel
    {
        real local_sum = 0.0;

        #pragma omp for schedule(static) nowait
        for (ui64 i = 0; i < num_simulations; i += simd_size) {
            // Loading of precalculated random number in a batch
            batch Z = xsimd::load_unaligned(&precomputed_random[i]);

            // Computation of ST with SIMD batch
            batch ST = batch_S0 * xsimd::exp(xsimd::fma(batch_T_sqrt, Z, batch_temp_val));

            // Computation of payoff
            batch payoff = xsimd::max(ST - batch_K, batch_zero);

            // Reduction on local_sum
            local_sum += xsimd::reduce_add(payoff);
        }

        // Atomic reducion on sum_payoffs
        #pragma omp atomic
        sum_payoffs += local_sum;
    }
    // free memory
    delete[] precomputed_random;

    return expRT * (sum_payoffs * inv_num_sim);
}


int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }
    // Set num of threads used (max available)
    omp_set_num_threads(omp_get_max_threads());

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs = std::stoull(argv[2]);

    // Input parameters
    ui64 S0 = 100;      // Initial stock price
    ui64 K = 110;       // Strike price
    real T = 1.0;     // Time to maturity (1 year)
    real r = 0.06;    // Risk-free interest rate
    real sigma = 0.2; // Volatility
    real q = 0.03;    // Dividend yield
    // Calculate constants
    real expRT = exp(-r * T); 
    real T_sqrt = sigma * real_sqrt(T);
    real temp_val = (r - q - 0.5 * sigma * sigma) * T;
    real inv_num_sim =(1.f / num_simulations);

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] << std::endl;

    real sum=0.0;

    double t1=dml_micros();
    // Parallel area with dynamic schedule and reduction (add) on sum 
    #pragma omp parallel for schedule(dynamic) reduction(+:sum)
    for (ui64 run = 0; run < num_runs; ++run) {

        sum+= black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations, expRT, T_sqrt, temp_val, inv_num_sim);

    }

    double t2=dml_micros();

    std::cout << std::fixed << std::setprecision(6) << " value= " << sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;

    return 0;
}