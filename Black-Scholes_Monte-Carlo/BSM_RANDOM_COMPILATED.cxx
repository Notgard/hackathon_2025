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
#include <vector>
#include <limits>
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <iomanip> // For setting precision
#include <mpi.h>
#include <boost/random.hpp>
#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#define ui64 u_int64_t
#ifndef REAL
#define REAL double
#endif  
#ifndef SEED
#define SEED 42
#endif
using real = REAL;


#include <sys/time.h>
double
dml_micros()
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
inline real real_exp(real r)
{
    if constexpr (sizeof(real) == sizeof(double)) return exp(r);
    if constexpr (sizeof(real) == sizeof(float)) return expf(r);
    if constexpr (sizeof(real) == sizeof(long double)) return expl(r);
}
// Function to calculate the Black-Scholes call option price using Monte Carlo method
real black_scholes_monte_carlo(ui64 S0, ui64 K, real T, real r, real T_sqrt, real temp_val, ui64 num_simulations)
{
    real sum_payoffs = 0.0;
    // random numbers generation with SEED set at compilation
    boost::mt19937 rng(SEED); 
    boost::normal_distribution<real> dist(0.0, 1.0);
    boost::variate_generator<boost::mt19937, boost::normal_distribution<real>> gen(rng, dist);

    // Computation of results
    for (ui64 i = 0; i < num_simulations; ++i)
    {
        real Z = gen();
        
        real ST = S0 * exp(temp_val + T_sqrt * Z);

        sum_payoffs += std::max<real>(ST - K, 0.0);
    }
    return exp(-r * T) * (sum_payoffs / num_simulations);
}

int main(int argc, char *argv[])
{
    
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <num_simulations> <num_runs>" << std::endl;
        return 1;
    }

    // Initialization of MPI
    MPI_Init(NULL,NULL);
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

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
    real T_sqrt = sigma * real_sqrt(T);
    real temp_val = (r - q - static_cast<real>(0.5) * sigma * sigma) * T;
    
    double t1,t2;
    

    real sum=0.0;
    // Start timer only for rank 0 MPI thread
    if(rank == 0){
        std::cout << "Global initial seed: " << SEED << "      argv[1]= " << argv[1] << "     argv[2]= " << argv[2] << std::endl;
        t1 = dml_micros();    
    }

    // Calculate chunck size and the remainder
    uint chunk_size = num_runs / size;
    int remainder = num_runs % size;
    
    if (rank < remainder) {
        chunk_size++;
    }
    
    // Computation of Monte Carlo black Scholes function
    for (ui64 run = 0; run < chunk_size; ++run) {

        sum+= black_scholes_monte_carlo(S0, K, T, r, T_sqrt, temp_val, num_simulations);

    }

    real total_sum = 0.0;
    // Reduction on global sum
    MPI_Reduce(&sum, &total_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);  
    
    MPI_Barrier(MPI_COMM_WORLD);
    // Stop timer only for rank 0 MPI thread
    if(rank== 0){
        t2=dml_micros();
        std::cout << std::fixed << std::setprecision(6) << " value= " << total_sum/num_runs << " in " << (t2-t1)/1000000.0 << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}
