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
#include <iomanip>         // For setting precision
#include <xsimd/xsimd.hpp> // Include xsimd library for optimized mathematical operations
#include <omp.h>
#include "pcg_random.hpp"

#include <armpl.h>

#define ui64 u_int64_t
#ifndef REAL
#define REAL double
#endif
using real = REAL;

#define assert_message(cond, msg)      \
    if (!(cond))                       \
    {                                  \
        std::cerr << msg << std::endl; \
        abort();                       \
    }

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
    if constexpr (sizeof(real) == sizeof(double))
        return sqrt(r);
    if constexpr (sizeof(real) == sizeof(float))
        return sqrtf(r);
    if constexpr (sizeof(real) == sizeof(long double))
        return sqrtl(r);
}

void arm_random_number_generator(real *random_numbers, size_t size, VSLStreamStatePtr &rng)
{
    int errcode = vsRngGaussian(VSL_RNG_METHOD_GAUSSIAN_BOXMULLER, rng, size, random_numbers, 0.0f, 1.0f);
}

real black_scholes_monte_carlo(ui64 S0, ui64 K, real T, real r, real sigma, real q, ui64 num_simulations, real expRT, real T_sqrt, real temp_val, real inv_num_sim)
{
    using batch = xsimd::batch<real>; // Batch SIMD
    constexpr size_t simd_size = batch::size;

    const batch batch_temp_val(temp_val);
    const batch batch_T_sqrt(T_sqrt);
    const batch batch_S0(S0);
    const batch batch_K(K);
    const batch batch_zero(0.0);

    real sum_payoffs = 0.0;

    ui64 chunk_size = simd_size * 100;

    VSLStreamStatePtr rng;
    const int seed = 1;
    int status = vslNewStream(&rng, VSL_BRNG_MT19937, seed);

    #pragma omp parallel
    {
        // Allocate local storage for random numbers
        real random_numbers[chunk_size];

        // std::cout << chunk_size << std::endl;

        real local_sum = 0.0;

        #pragma omp for schedule(static) nowait
        for (ui64 i = 0; i < num_simulations; i += chunk_size)
        {

            // Generate a chunk of random numbers
            ui64 current_chunk_size = std::min(chunk_size, num_simulations - i);
            arm_random_number_generator(random_numbers, current_chunk_size, rng);

            for (ui64 j = 0; j < current_chunk_size; j += simd_size)
            {
                // Load a batch of random numbers
                batch Z = xsimd::load_unaligned(&random_numbers[j]);

                // Compute ST in a batch SIMD
                batch ST = batch_S0 * xsimd::exp(xsimd::fma(batch_T_sqrt, Z, batch_temp_val));

                // Compute payoffs
                batch payoff = xsimd::max(ST - batch_K, batch_zero);

                // Accumulate locally
                local_sum += xsimd::reduce_add(payoff);
            }
        }
        // Atomic accumulation for the global sum
        #pragma omp atomic
        sum_payoffs += local_sum;

    }
    
    vslDeleteStream(&rng);

    return expRT * (sum_payoffs * inv_num_sim);
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }
    omp_set_num_threads(omp_get_max_threads());

    ui64 num_simulations = std::stoull(argv[1]);
    ui64 num_runs = std::stoull(argv[2]);

    // Input parameters
    ui64 S0 = 100;                   // Initial stock price
    ui64 K = 110;                    // Strike price
    real T = 1.0;                    // Time to maturity (1 year)
    real r = 0.06;                   // Risk-free interest rate
    real sigma = 0.2;                // Volatility
    real q = 0.03;                   // Dividend yield
    real expRT = xsimd::exp(-r * T); // Use xsimd::exp for optimized computation
    real T_sqrt = sigma * xsimd::sqrt(T);
    real temp_val = (r - q - 0.5 * sigma * sigma) * T;
    real inv_num_sim = (1.f / num_simulations);

    std::cout << "Global initial seed: " << global_seed << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] << std::endl;

    real sum = 0.0;

    double t1 = dml_micros();

    #pragma omp parallel for schedule(dynamic) reduction(+ : sum)
    for (ui64 run = 0; run < num_runs; ++run)
    {

        sum += black_scholes_monte_carlo(S0, K, T, r, sigma, q, num_simulations, expRT, T_sqrt, temp_val, inv_num_sim);
    }

    double t2 = dml_micros();

    std::cout << std::fixed << std::setprecision(6) << " value= " << sum / num_runs << " in " << (t2 - t1) / 1000000.0 << " seconds" << std::endl;

    return 0;
}