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
#include "random.h"

#define M 100000000  // Total number of throws
#define N 100        // Number of blocks

using namespace std;

void RND_Initialize(Random&); // Function to initialize the pseudorandom number generator
double error(vector<double>&, vector<double>&, int); // Function for statistical uncertainty estimation

double f(double); // Function to integrate with uniform sampling
double P(double); // Probability density function for importance sampling
double g(double); // Function to integrate with importance sampling
 
int main (int argc, char *argv[]){
    
    Random rnd;
    RND_Initialize(rnd); // Initialize the random number generator
    
    // exercise 02.1
    int L = static_cast<int>(M/N);    // Number of throws in each block
    vector<double> ave(N,0.), av2(N,0.), sum_prog(N,0.), su2_prog(N,0.), err_prog(N,0.);
    
    // Compute the integral sampling a uniform distribution
    double xmin = 0.;
    double xmax = 1.;
    
    // Loop over each block
    for(int i=0; i<N; i++){
        double sum = 0.;
        // Loop over throws in each block
        for(int j=0; j<L; j++){
            sum += f(rnd.Rannyu(xmin,xmax)); // Accumulate measures
        }
        ave[i] = sum/L*(xmax-xmin);   // Estimate in each block
        av2[i] = pow(ave[i],2);
    }
    
    // Calculate cumulative average and statistical uncertainty
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += ave[j];   // Cumulative sum of estimates
            su2_prog[i] += av2[j];   // Cumulative sum of squared estimates
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outInt;
    outInt.open("int_result.txt");
    if (outInt.is_open()){
        for(int i=0; i<N; i++){
            outInt << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open int_result.txt" << endl;
    outInt.close();
    
    // Reset vectors for importance sampling
    ave.assign(ave.size(),0.);
    av2.assign(av2.size(),0.);
    sum_prog.assign(sum_prog.size(),0.);
    su2_prog.assign(su2_prog.size(),0.);
    err_prog.assign(err_prog.size(),0.);
    
    // Compute the integral using importance sampling
    double x = 0.;
    double r = 0.;
    double Pmax = 3/2.; // Maximum value of P(x) for normalization
    
    // Loop over each block
    for(int i=0; i<N; i++){
        double sum = 0.;
        // Loop over throws in each block
        for(int j=0; j<L; j++){
            // Use accept/reject method to sample values distributed as P(x)
            do{
                x = rnd.Rannyu(xmin,xmax);
                r = rnd.Rannyu();
            } while(r >= P(x)/Pmax);
            sum += g(x); // Accumulate measures
        }
        ave[i] = sum/L;   // Estimate in each block
        av2[i] = pow(ave[i],2);
    }
    
    // Calculate cumulative average and statistical uncertainty
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += ave[j];   // Cumulative sum of estimates
            su2_prog[i] += av2[j];   // Cumulative sum of squared estimates
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outIS;
    outIS.open("int_IS_result.txt");
    if (outIS.is_open()){
        for(int i=0; i<N; i++){
            outIS << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open int_IS_result.txt" << endl;
    outIS.close();
    
    // exercise 02.2
    int nstep = 100; // Number of steps in the random walk
    double a = 1.; // Step length
    
    // 3D Random Walks on a cubic lattice
    vector<double> pos_mean(nstep,0.), po2_mean(nstep,0.), err(nstep,0.);
    
    // Loop over each block
    for(int i=0; i<N; i++){
        vector<double> dist(nstep, 0.); // Distance for each step
        // Loop over random walks in each block
        for(int j=0; j<L; j++){
            double X = 0., Y = 0., Z = 0.; // Starting at origin
            // Loop over each step
            for(int k=0; k<nstep; k++){
                double r = rnd.Rannyu(-1.5, 1.5);
                double sign = rnd.Rannyu();
                sign = (sign < 0.5) ? -1. : 1.;
                // Move in X, Y, or Z direction
                if (r<-0.5) X += sign*a;
                else if (r>=-0.5 && r<0.5) Y += sign*a;
                else if (r>=0.5) Z += sign*a;
                // Accumulate squared distance
                dist[k] += pow(X,2) + pow(Y,2) + pow(Z,2);
            }
        }
        
        // Update mean and squared mean for each step
        for(int k=0; k<nstep; k++){
            pos_mean[k] += sqrt(dist[k]/L); // Average distance
            po2_mean[k] += pow(sqrt(dist[k]/L),2); // Squared average distance
        }
    }
    
    // Calculate cumulative average and statistical uncertainty
    for(int k=0; k<nstep; k++){
        pos_mean[k] /= N;
        po2_mean[k] /= N;
        err[k] = error(pos_mean, po2_mean, k); // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outRWdisc;
    outRWdisc.open("RW_disc_result.txt");
    if (outRWdisc.is_open()){
        for(int i=0; i<nstep; i++){
            outRWdisc << pos_mean[i] << " " << err[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open RW_disc_result.txt" << endl;
    outRWdisc.close();
    
    // 3D Random Walks in the continuum
    pos_mean.assign(pos_mean.size(),0.);
    po2_mean.assign(po2_mean.size(),0.);
    err.assign(err.size(),0.);
    
    // Loop over each block
    for(int i=0; i<N; i++){
        vector<double> dist(nstep, 0.); // Distance for each step
        // Loop over random walks in each block
        for(int j=0; j<L; j++){
            double X = 0., Y = 0., Z = 0.; // Starting at origin
            // Loop over each step
            for(int k=0; k<nstep; k++){
                // Generate random angles for direction
                double theta = acos(1-2*rnd.Rannyu());
                double phi = 2*M_PI*rnd.Rannyu();
                // Update position
                X += a*sin(theta)*cos(phi);
                Y += a*sin(theta)*sin(phi);
                Z += a*cos(theta);
                // Accumulate squared distance
                dist[k] += pow(X,2) + pow(Y,2) + pow(Z,2);
            }
        }
        
        // Update mean and squared mean for each step
        for(int k=0; k<nstep; k++){
            pos_mean[k] += sqrt(dist[k]/L); // Average distance
            po2_mean[k] += pow(sqrt(dist[k]/L),2); // Squared average distance
        }
    }
    
    // Calculate cumulative average and statistical uncertainty
    for(int k=0; k<nstep; k++){
        pos_mean[k] /= N;
        po2_mean[k] /= N;
        err[k] = error(pos_mean, po2_mean, k); // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outRW;
    outRW.open("RW_result.txt");
    if (outRW.is_open()){
        for(int i=0; i<nstep; i++){
            outRW << pos_mean[i] << " " << err[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open RW_result.txt" << endl;
    outRW.close();
    
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

double error(vector<double>&ave, vector<double>&av2, int n){
    if (n==0) return 0;
    else return sqrt((av2[n] - pow(ave[n],2))/n);
}

double f(double x) {return M_PI/2. * cos(M_PI/2.*x);}

double P(double x) {return 3/2.*(1-pow(x,2));}

double g(double x) {return M_PI/3.*cos(M_PI*x/2.)/(1-pow(x,2));}

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
