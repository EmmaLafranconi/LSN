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
#include <armadillo>
#include "random.h"

#define nblocks 100    // Number of blocks
#define nsteps 10000   // Number of steps

using namespace std;
using namespace arma;

void RND_Initialize(Random&); // Function to initialize the pseudorandom number generator
double error(vec&, vec&, int); // Function for statistical uncertainty estimation
double Vpot(double); // Function for the potential
double psi(double, double, double); // Trial wave function
double Energy(double, double, double); // Energy expectation value function
mat Hamiltonian(Random&, double, double, double, bool, bool); // Function to compute Hamiltonian expectation value
 
int main (int argc, char *argv[]){
    
    Random rnd;
    RND_Initialize(rnd); // Initialize the random number generator

    // exercise 08.1
    // Parameters for Variational Monte Carlo
    double mu = 0.85;
    double sigma = 0.68;
    double delta = 2.9;
    bool acceptance = true; // Enable acceptance logging
    bool configurations = false; // Disable configurations logging
    
    // Compute Hamiltonian expectation values
    mat Hmean = Hamiltonian(rnd,mu,sigma,delta,acceptance,configurations);
    
    // Write results to file
    ofstream output;
    output.open("VMC_result.txt");
    if (output.is_open()){
        for(int i=0; i<nblocks; i++){
            output << Hmean(i,0) << " " << Hmean(i,1) << endl;
        }
    } else cerr << "PROBLEM: Unable to open VMC_result.txt" << endl;
    output.close();

    // exercise 08.2
    mu = 2.; // Starting point for variational parameter mu
    sigma = 2.; // Starting point for variational parameter sigma
    double chi = 0.; // Initialize chi parameter for convergence check
    int beta = 1; // Initialize beta parameter for SA temperature schedule
    acceptance = false; // Disable acceptance logging for Simulated Annealing
    rowvec H = Hamiltonian(rnd,mu,sigma,delta,acceptance,configurations).row(nblocks-1); // Get last row of Hamiltonian values
    
    // Open file for writing SA configurations
    ofstream out("SA_configurations.txt");
    out << "#      BETA:           MU:        SIGMA:       ENERGY:        ERROR:" << endl;
    out.close();
    
    // Simulated Annealing optimization loop
    do{
        double deltaSA = pow(double(beta),-0.5); // Temperature scaling factor
        // Perform 5 Metropolis steps at each temperature (beta)
        for(int j=0; j<5; j++){
            // Generate new candidate parameters for mu and sigma
            double mu_new = rnd.Rannyu(mu-deltaSA,mu+deltaSA);
            double sigma_new = rnd.Rannyu(sigma-deltaSA,sigma+deltaSA);
            // Compute Hamiltonian values for the new parameters
            rowvec Hnew = Hamiltonian(rnd,mu_new,sigma_new,delta,acceptance,configurations).row(nblocks-1);
            // Calculate acceptance probability based on energy difference
            double acceptance = min(1.,exp(-beta*(Hnew(0)-H(0))));
            // Metropolis acceptance step
            if(rnd.Rannyu() <= acceptance){
                chi = abs(Hnew(0)-H(0))/sqrt(pow(Hnew(1),2)+pow(H(1),2)); // Calculate compatibility
                mu = mu_new;
                sigma = sigma_new;
                H = Hnew;
            };
            // Write SA configurations to file
            out.open("SA_configurations.txt",ios::app);
            out << setw(12) << beta
                << setw(14) << mu << setw(14) << sigma
                << setw(14) << H(0) << setw(14) << H(1) << endl;
            out.close();
        }
        // Write SA results to file
        ofstream outt;
        outt.open("SA_result.txt",ios::app);
        outt << beta << " " << H(0) << " " << H(1) << endl;
        outt.close();
        beta+=2; // Increase beta for next iteration
    } while(chi>1.); // Stop optimization if energy values are compatible, you can't do better!
    
    // fix optimal parameters found with Simulated Annealing
    mu = -0.803538;
    sigma = 0.613555;
    delta = 2.6;
    acceptance = true; // Enable acceptance logging
    configurations = true; // Enable configurations logging
    
    // Compute Hamiltonian expectation values
    Hmean = Hamiltonian(rnd,mu,sigma,delta,acceptance,configurations);
    
    // Write results to file
    ofstream outputt;
    outputt.open("VMC_result2.txt");
    if (outputt.is_open()){
        for(int i=0; i<nblocks; i++){
            outputt << Hmean(i,0) << " " << Hmean(i,1) << endl;
        }
    } else cerr << "PROBLEM: Unable to open VMC_result2.txt" << endl;
    outputt.close();
    
    rnd.SaveSeed(); // Save the state of the random number generator

    return 0;
}

void RND_Initialize(Random& rnd){
    int seed[4];
    int p1, p2;
    ifstream Primes("Primes");
    if (Primes.is_open()){
       Primes >> p1 >> p2 ;
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

double error(vec&ave, vec&av2, int n){
    if (n==0) return 0;
    else return sqrt((av2(n) - pow(ave(n),2))/n);
}

double Vpot(double x){
    return pow(x,4) - 5./2.*pow(x,2);
}

double psi(double x, double mu, double sigma){
    double alpha = pow((x-mu)/sigma,2);
    double beta = pow((x+mu)/sigma,2);
    return exp(-alpha/2.) + exp(-beta/2.);
}

double Energy(double x, double mu, double sigma){
    double alpha = pow((x-mu)/sigma,2);
    double beta = pow((x+mu)/sigma,2);
    double kenergy = - 0.5/pow(sigma,2) * (alpha*exp(-alpha/2.) + beta*exp(-beta/2.) - exp(-alpha/2.) - exp(-beta/2.));
    return kenergy/psi(x,mu,sigma) + Vpot(x);
}

mat Hamiltonian(Random &rnd, double mu, double sigma, double delta, bool accept, bool config){
    mat H(nblocks,2); // Matrix to store Hamiltonian values and errors
    vec ave = zeros<vec>(nblocks),
        av2 = zeros<vec>(nblocks),
        sum_prog = zeros<vec>(nblocks),
        su2_prog = zeros<vec>(nblocks),
        err_prog = zeros<vec>(nblocks);
    
    // If acceptance logging is enabled, open a file to store acceptance rates
    if(accept){
        ofstream out("acceptance.txt");
        out << "#   N_BLOCK:   ACCEPTANCE:" << endl;
        out.close();
    }
    
    double x = 0.; // Starting point for Metropolis algorithm
    for(int i=0; i<nblocks; i++){
        double nattempts = 0.; // Counter for Metropolis attempts
        double naccepted = 0.; // Counter for accepted Metropolis moves
        double sum = 0.;
        for(int j=0; j<nsteps; j++){
            double xnew = rnd.Rannyu(x-delta,x+delta);
            double acceptance = min(1.,pow(psi(xnew,mu,sigma),2)/pow(psi(x,mu,sigma),2));
            if(rnd.Rannyu() <= acceptance){ // Metropolis acceptance evaluation
                x = xnew;
                naccepted++;
            }
            nattempts++;
            sum += Energy(x,mu,sigma); // Accumulate energy measures
            
            // If configuration logging is enabled, append current position to file
            if(config){
                ofstream outt;
                outt.open("VMC_configurations.txt",ios::app);
                outt << x << endl;
                outt.close();
            }
        }
        // Calculate average energy and squared average energy in each block
        ave(i) = sum/nsteps;
        av2(i) = pow(ave(i),2);
        
        // If acceptance logging is enabled, calculate acceptance fraction and log it
        if(accept){
            double fraction = 0.;
            ofstream out;
            out.open("acceptance.txt",ios::app);
            if(nattempts > 0) fraction = double(naccepted)/double(nattempts);
            else fraction = 0.;
            out << setw(12) << i << setw(14) << fraction << endl;
            out.close();
        }
    }
    
    // Compute cumulative averages and statistical uncertainties
    for(int i=0; i<nblocks; i++){
        for(int j=0; j<i+1; j++){
            sum_prog(i) += ave(j);
            su2_prog(i) += av2(j);
        }
        sum_prog(i)/=(i+1);  // Cumulative average
        su2_prog(i)/=(i+1);  // Cumulative square average
        err_prog(i) = error(sum_prog, su2_prog, i);   // Statistical uncertainty
        // Store cumulative average and uncertainty in the matrix H
        H(i,0) = sum_prog(i);
        H(i,1) = err_prog(i);
    }
    
    return H;
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
