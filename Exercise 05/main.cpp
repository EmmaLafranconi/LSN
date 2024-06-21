/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/

// istruzioni per compilare

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include "random.h"

using namespace std;

void RND_Initialize(Random&); // Function to initialize the pseudorandom number generator
double error(vector<double>&, vector<double>&, int); // Function for statistical uncertainty estimation
double r(double, double, double); // Function for the radius
double p_gs(double, double, double); // Ground state probability density
double p_2p(double, double, double); // 2p state probability density
 
int main (int argc, char *argv[]){
    
    Random rnd;
    RND_Initialize(rnd); // Initialize the random number generator
    
    // Read input parameters from file "input.dat"
    ifstream in("input.dat");
    string prop;
    double x, y, z, delta, pdensity, sim_type;
    int nblocks, nsteps;
    while ( !in.eof() ){
        in >> prop;
        if( prop == "PROBABILITY_DENSITY"){
            in >> pdensity;
            if(pdensity > 1){
                cerr << "PROBLEM: unknown probability density" << endl;
                exit(EXIT_FAILURE);
            }
            if(pdensity == 0) cout << "GROUND STATE probability density" << endl;
            else if(pdensity == 1) cout << "2P STATE probability densityY" << endl;
        } else if( prop == "TRANSITION_PROBABILITY"){
            in >> sim_type;
            if(sim_type > 1){
                cerr << "PROBLEM: unknown transition probability" << endl;
                exit(EXIT_FAILURE);
            }
            if(sim_type == 0) cout << "UNIFORM transition probability" << endl;
            else if(sim_type == 1) cout << "GAUSSIAN transition probability" << endl;
        } else if( prop == "X_STARTING_POINT" ) in >> x;
          else if( prop == "Y_STARTING_POINT" ) in >> y;
          else if( prop == "Z_STARTING_POINT" ) in >> z;
          else if( prop == "DELTA" ) in >> delta;
          else if( prop == "NBLOCKS" ) in >> nblocks;
          else if( prop == "NSTEPS" ) in >> nsteps;
          else if( prop == "ENDINPUT" ) break;
          else cerr << "PROBLEM: unknown input" << endl;
    }
    in.close();
    
    // Initialize vectors and variables for averaging and acceptance calculation
    vector<double> ave(nblocks,0.), av2(nblocks,0.), sum_prog(nblocks,0.), su2_prog(nblocks,0.), err_prog(nblocks,0.);
    double xnew, ynew, znew, acceptance;
    
    // Open file for writing acceptance rate
    ofstream out("acceptance.txt");
    out << "#   N_BLOCK:   ACCEPTANCE:" << endl;
    out.close();
    
    // Metropolis algorithm for sampling
    for(int i=0; i<nblocks; i++){
        double nattempts = 0.;
        double naccepted = 0.;
        double sum = 0.;
        for(int j=0; j<nsteps; j++){
            // Generate trial moves based on transition probability type
            if(sim_type == 0){ // Uniform transition probability
                xnew = rnd.Rannyu(x-delta,x+delta);
                ynew = rnd.Rannyu(y-delta,y+delta);
                znew = rnd.Rannyu(z-delta,z+delta);
            } else if(sim_type == 1){ // Gaussian transition probability
                xnew = rnd.Gauss(x,delta);
                ynew = rnd.Gauss(y,delta);
                znew = rnd.Gauss(z,delta);
            }
            // Calculate acceptance probability based on probability density
            if(pdensity == 0) acceptance = min(1.,p_gs(xnew,ynew,znew)/p_gs(x,y,z)); // Ground State probability density
            else if(pdensity == 1) acceptance = min(1.,p_2p(xnew,ynew,znew)/p_2p(x,y,z)); // 2p State probability density
            // Metropolis acceptance step
            if(rnd.Rannyu() <= acceptance){
                x = xnew;
                y = ynew;
                z = znew;
                naccepted++;
            }
            nattempts++;
            sum += r(x,y,z); // Accumulate measures
            
            // Output coordinates to file
            ofstream outt;
            outt.open("coordinates.txt",ios::app);
            outt << x << setw(14) << y << setw(14) << z << setw(14) << r(x,y,z) << endl;
            outt.close();
        }
        // Calculate average radius in each block
        ave[i] = sum/nsteps;
        av2[i] = pow(ave[i],2);
        
        // Calculate acceptance fraction and write to file
        double fraction = 0.;
        out.open("acceptance.txt",ios::app);
        if(nattempts > 0) fraction = double(naccepted)/double(nattempts);
        else fraction = 0.;
        out << setw(12) << i << setw(14) << fraction << endl;
        out.close();
    }
    
    // Calculate cumulative averages and statistical uncertainties
    for(int i=0; i<nblocks; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += ave[j];
            su2_prog[i] += av2[j];
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream output;
    output.open("radius.txt");
    if (output.is_open()){
        for(int i=0; i<nblocks; i++){
            output << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open radius.txt" << endl;
    output.close();
    
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

double r(double x, double y, double z){
    return sqrt(x*x + y*y + z*z);
}

double p_gs(double x, double y, double z){
    return pow(abs(exp(-r(x,y,z))/sqrt(M_PI)),2);
}

double p_2p(double x, double y, double z){
    return pow(abs(1./8.*sqrt(2./M_PI) * exp(-r(x,y,z)/2.) * z),2);
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
