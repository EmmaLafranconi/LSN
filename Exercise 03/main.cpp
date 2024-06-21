/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include "random.h"

#define M 1000000   // Total number of throws
#define N 100       // Number of blocks

using namespace std;

void RND_Initialize(Random&, int); // Function to initialize the pseudorandom number generator
double error(vector<double>&, vector<double>&, int); // Function for statistical uncertainty estimation

int main (int argc, char *argv[]){
    
    Random rnd;
    int rank = 10; // Parameter for the random number generator
    RND_Initialize(rnd,rank); // Initialize the random number generator
    
    // exercise 03.1
    int L = static_cast<int>(M/N);    // Number of throws in each block
    vector<double> aveC(N,0.), av2C(N,0.), sum_progC(N,0.), su2_progC(N,0.), err_progC(N,0.);
    vector<double> aveP(N,0.), av2P(N,0.), sum_progP(N,0.), su2_progP(N,0.), err_progP(N,0.);
    
    double S0 = 100.;       // asset price at t=0
    double K = 100.;        // strike price
    double T = 1.;          // delivery time
    double r = 0.1;         // risk-free interest rate
    double sigma = 0.25;    // volatility
    
    // sampling directly the final asset price S(T) for a GBM(r, sigma^2)
    for(int i=0; i<N; i++){
        double ST = 0;
        double C = 0.;
        double P = 0.;
        
        for(int j=0; j<L; j++){
            ST = S0 * exp((r-pow(sigma,2)/2.)*T + sigma*rnd.Gauss(0.,T));
            C += exp(-r*T)*max(0.,ST-K); // Accumulate measures for European call-option price
            P += exp(-r*T)*max(0.,K-ST); // Accumulate measures for European put-option price
        }
        // Estimate in each block
        aveC[i] = C/L;
        aveP[i] = P/L;
        av2C[i] = pow(aveC[i],2);
        av2P[i] = pow(aveP[i],2);
    }
    
    // Calculate cumulative averages and statistical uncertainties
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_progC[i] += aveC[j];
            sum_progP[i] += aveP[j];
            su2_progC[i] += av2C[j];
            su2_progP[i] += av2P[j];
        }
        sum_progC[i]/=(i+1);  // Cumulative average
        sum_progP[i]/=(i+1);
        su2_progC[i]/=(i+1);  // Cumulative square average
        su2_progP[i]/=(i+1);
        err_progC[i] = error(sum_progC, su2_progC, i);   // Statistical uncertainty
        err_progP[i] = error(sum_progP, su2_progP, i);
    }
    
    // Write results to file
    ofstream outGBM;
    outGBM.open("GBM_result.txt");
    if (outGBM.is_open()){
        for(int i=0; i<N; i++){
            outGBM << sum_progC[i] << " " << err_progC[i] << " " << sum_progP[i] << " " << err_progP[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open GBM_result.txt" << endl;
    outGBM.close();
    
    // Reset vectors for discrete sampling
    aveC.assign(aveC.size(),0.);
    av2C.assign(av2C.size(),0.);
    aveP.assign(aveP.size(),0.);
    av2P.assign(av2P.size(),0.);
    sum_progC.assign(sum_progC.size(),0.);
    su2_progC.assign(su2_progC.size(),0.);
    err_progC.assign(err_progC.size(),0.);
    sum_progP.assign(sum_progP.size(),0.);
    su2_progP.assign(su2_progP.size(),0.);
    err_progP.assign(err_progP.size(),0.);
    
    vector<double> St(100,0.); // Vector to store asset price path
    
    // sampling the discretized GBM(r, sigma^2) path of the asset price dividing [0,T] in 100 time intervals
    for(int i=0; i<N; i++){
        double ST = 0;
        double C = 0.;
        double P = 0.;
        
        for(int j=0; j<L; j++){
            for(int t=1; t<=100; t++){
                St[0] = S0;
                St[t] = St[t-1] * exp((r-pow(sigma,2)/2.)*T/100. + sigma*rnd.Gauss(0.,1.)*sqrt(T/100.));
            }
            ST = St[100];
            C += exp(-r*T)*max(0.,ST-K);    // Accumulate measures
            P += exp(-r*T)*max(0.,K-ST);
        }
        // Estimate in each block
        aveC[i] = C/L;
        aveP[i] = P/L;
        av2C[i] = pow(aveC[i],2);
        av2P[i] = pow(aveP[i],2);
    }
    
    // Calculate cumulative averages and statistical uncertainties for discrete sampling
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_progC[i] += aveC[j];
            sum_progP[i] += aveP[j];
            su2_progC[i] += av2C[j];
            su2_progP[i] += av2P[j];
        }
        sum_progC[i]/=(i+1);  // Cumulative average
        sum_progP[i]/=(i+1);
        su2_progC[i]/=(i+1);  // Cumulative square average
        su2_progP[i]/=(i+1);
        err_progC[i] = error(sum_progC, su2_progC, i);   // Statistical uncertainty
        err_progP[i] = error(sum_progP, su2_progP, i);
    }
    
    // Write results to file
    ofstream outGBMdisc;
    outGBMdisc.open("GBM_disc_result.txt");
    if (outGBMdisc.is_open()){
        for(int i=0; i<N; i++){
            outGBMdisc << sum_progC[i] << " " << err_progC[i] << " " << sum_progP[i] << " " << err_progP[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open GBM_disc_result.txt" << endl;
    outGBMdisc.close();
    
    rnd.SaveSeed(); // Save the state of the random number generator
    
    return 0;
}

void RND_Initialize(Random& rnd, int rank){
    int seed[4];
    int p1, p2;
    ifstream Primes("Primes");
    if (Primes.is_open()){
        for(int i=0; i<rank; i++){
            Primes >> p1 >> p2 ;
        }
    } else cerr << "PROBLEM: Unable to open Primes" << endl;
    Primes.close();

    ifstream input("seed.in");
    string property;
    if (input.is_open()){
       while ( !input.eof() ){
          input >> property;
          if( property == "RANDOMSEED" ){
             input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
             rnd.SetRandom(seed,p1,p2);
          }
       }
       input.close();
    } else cerr << "PROBLEM: Unable to open seed.in" << endl;
}

double error(vector<double>&ave, vector<double>&av2, int n){
    if (n==0) return 0;
    else return sqrt((av2[n] - pow(ave[n],2))/n);
}

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
