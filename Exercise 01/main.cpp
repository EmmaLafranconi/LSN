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

#define M 10000  // Total number of throws
#define N 100    // Number of blocks

using namespace std;

void RND_Initialize(Random&); // Function to initialize the pseudorandom number generator
double error(vector<double>&, vector<double>&, int); // Function for statistical uncertainty estimation
 
int main (int argc, char *argv[]){
    
    Random rnd;
    RND_Initialize(rnd); // Initialize the random number generator

    // exercise 01.1
    int L = static_cast<int>(M/N);    // Number of throws in each block
    // Vectors to store average, square of average, progressive sum, progressive square sum, and progressive error
    vector<double> ave(N,0.), av2(N,0.), sum_prog(N,0.), su2_prog(N,0.), err_prog(N,0.);
    
    // Computing the estimation of the mean and its uncertainty
    for(int i=0; i<N; i++){
        double sum = 0.;
        for(int j=0; j<L; j++){
            sum += rnd.Rannyu(); // Accumulate measures
        }
        ave[i] = sum/L;          // Average of the block
        av2[i] = pow(ave[i],2);  // Square of the average
    }
    
    // Calculate progressive mean and its error
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += ave[j];   // Sum of averages
            su2_prog[i] += av2[j];   // Sum of squares of averages
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outMean;
    outMean.open("mean_result.txt");
    if (outMean.is_open()){
        for(int i=0; i<N; i++){
            outMean << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open mean_result.txt" << endl;
    outMean.close();
    
    // Reset vectors for variance calculation
    ave.assign(ave.size(),0.);
    av2.assign(av2.size(),0.);
    sum_prog.assign(sum_prog.size(),0.);
    su2_prog.assign(su2_prog.size(),0.);
    err_prog.assign(err_prog.size(),0.);
    
    // Computing the estimation of the variance and its uncertainty
    for(int i=0; i<N; i++){
        double sum = 0.;
        for(int j=0; j<L; j++){
            sum += pow(rnd.Rannyu()-0.5,2); // Calculate squared deviation from the mean (0.5)
        }
        ave[i] = sum/L;          // Average of squared deviations (variance)
        av2[i] = pow(ave[i],2);  // Square of the average
    }
    
    // Calculate progressive variance and its error
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += ave[j];   // Sum of variances
            su2_prog[i] += av2[j];   // Sum of squares of variances
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outVar;
    outVar.open("var_result.txt");
    if (outVar.is_open()){
        for(int i=0; i<N; i++){
            outVar << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open var_result.txt" << endl;
    outVar.close();
    
    // Computing the chi^2 test
    double a = 0.;
    double b = 1.;
    double ampl = (b-a)/double(N); // Width of each bin
    
    // Vectors to store chi^2 values and bin counts
    vector<double> chi2(100,0.);
    vector<unsigned int> bins(N,0);
    
    // Perform the chi^2 test
    for(int i=0; i<100; i++){
        bins.assign(bins.size(),0); // Reset bin counts for each test iteration
        
        for(int j=0; j<M; j++){
            // Binary search to find the correct bin for the random number
            int low = 0;
            int high = N-1;
            int bin_index = -1;
            double rand = rnd.Rannyu();
            
            do{
                int mid = (low + high)/2;
                double bin_low = double(a) + double(mid)*ampl;
                double bin_high = double(a) + double(mid+1)*ampl;
                
                // Check which bin the random number falls into
                if (rand >= bin_low && rand < bin_high) bin_index = mid;
                else if (rand < bin_low) high = mid-1;
                else low = mid+1;
            } while(low <= high && bin_index == -1);
            
            // Increment the count for the found bin
            if (bin_index != -1) {
                bins[bin_index]++;
            }
        }
        
        // Compute the chi^2 value for this iteration
        for(int k=0; k<N; k++){
            chi2[i] += pow(double(bins[k])-double(L),2.)/double(L);
        }
    }
    
    // Write results to file
    ofstream outChi2;
    outChi2.open("chi2_result.txt");
    if (outChi2.is_open()){
        for(int i=0; i<100; i++){
            outChi2 << chi2[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open chi2_result.txt" << endl;
    outChi2.close();
    
    // exercise 01.2
    double lambda = 1.; // Parameter for exponential distribution
    double mu = 0.;     // Mean for Lorentzian distribution
    double gamma = 1.;  // Width for Lorentzian distribution
    
    // Define the number of throws for each experiment
    vector<double> throws {1., 2., 10., 100.};
    
    // Vectors to store the sums for uniform, exponential, and Lorentzian distributions
    vector<double> sum_unif(M,0.), sum_exp(M,0.), sum_lor(M,0.);
    
    // Open files to store results
    ofstream outUnif("unif_result.txt");
    ofstream outExp("exp_result.txt");
    ofstream outLor("lor_result.txt");
    
    // Loop over the total number of experiments
    for(int i=0; i<M; i++){
        unsigned int k = 0; // Counter for the number of throws
        
        // Loop over different number of throws
        for(int j=0; j<throws.size(); j++){
            // Perform the specified number of throws for each distribution
            while(k < throws[j]){
                sum_unif[i] += rnd.Rannyu();         // Sum of uniform random numbers
                sum_exp[i] += rnd.Exp(lambda);       // Sum of exponential random numbers
                sum_lor[i] += rnd.Lorentz(mu,gamma); // Sum of Lorentzian random numbers
                k++;
            }
            // Write the averages to the corresponding output files
            outUnif << sum_unif[i]/throws[j] << " ";
            outExp << sum_exp[i]/throws[j] << " ";
            outLor << sum_lor[i]/throws[j] << " ";
        }
        outUnif << endl;
        outExp << endl;
        outLor << endl;
    }
    
    // Close the output files
    outUnif.close();
    outExp.close();
    outLor.close();
    
    // exercise 01.3
    L = static_cast<int>(1000000/N);    // Number of throws in each block
    
    // Vectors to store the estimated value of pi and its square
    vector<double> pi(N,0.), pi2(N,0.);
    sum_prog.assign(sum_prog.size(),0.);
    su2_prog.assign(su2_prog.size(),0.);
    err_prog.assign(err_prog.size(),0.);
    
    // Parameters for Buffon's needle experiment
    double d = 1.5; // Distance between lines
    double l = 1.; // Length of the needle
    
    // Loop over each block
    for(int i=0; i<N; i++){
        unsigned int Nhit = 0; // Number of hits where the needle crosses a line
        
        // Loop over the number of throws in each block
        for(int j=0; j<L; j++){
            // Generate a random position for the center of the needle
            double x0 = rnd.Rannyu(0.,d);
            
            // Generate a random orientation for the needle
            double x, y, cos_theta;
            do{
                x = rnd.Rannyu(-1.,1.);
                y = rnd.Rannyu(-1.,1.);
            } while(pow(x,2) + pow(y,2) >= 1); // Ensure x and y lie within a unit circle
            
            cos_theta = x/(sqrt(pow(x,2) + pow(y,2))); // Calculate the cosine of the angle
            
            // Check if the needle crosses a line
            if (x0 + l*cos_theta < 0 || x0 + l*cos_theta > d) Nhit++;
        }
        
        // Estimate pi for this block
        pi[i] = 2.*l*L/(Nhit*d);
        pi2[i] = pow(pi[i],2);
    }
    
    // Calculate the cumulative average and its error
    for(int i=0; i<N; i++){
        for(int j=0; j<i+1; j++){
            sum_prog[i] += pi[j];    // Cumulative sum of pi estimates
            su2_prog[i] += pi2[j];   // Cumulative sum of pi^2 estimates
        }
        sum_prog[i]/=(i+1);  // Cumulative average
        su2_prog[i]/=(i+1);  // Cumulative square average
        err_prog[i] = error(sum_prog, su2_prog, i);   // Statistical uncertainty
    }
    
    // Write results to file
    ofstream outBuffon;
    outBuffon.open("needle_result.txt");
    if (outBuffon.is_open()){
        for(int i=0; i<N; i++){
            outBuffon << sum_prog[i] << " " << err_prog[i] << endl;
        }
    } else cerr << "PROBLEM: Unable to open needle_result.txt" << endl;
    outBuffon.close();
    
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

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
