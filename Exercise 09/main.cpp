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
#include <filesystem>
#include "random.h"
#include "TSP.h"

using namespace std;
using namespace arma;
namespace fs = std::filesystem;

void RND_Initialize(Random&); // Function to initialize the pseudorandom number generator
 
int main (int argc, char *argv[]){
    if(argc < 2) {
        cerr << "Usage: " << argv[0] << " <circle|square>" << endl;
        return 1;
    }
    
    // Get the shape type from command line arguments
    string shape = argv[1];
    
    // Initialize the pseudorandom number generator
    Random rnd;
    RND_Initialize(rnd);
    
    // Define problem parameters
    int nCity = 34;           // Number of cities
    int PopSize = 300;        // Population size
    int nGen = 300;           // Number of generations
    double p = 2.0;           // Selection parameter
    double Pcross = 0.70;     // Crossover probability
    double Pmutate = 0.10;    // Mutation probability
    double Pshift = 0.10;     // Shift mutation probability
    double Ppermutate = 0.10; // Permutation probability
    double Pinvert = 0.10;    // Inversion mutation probability
    
    // Create a path with nCity cities
    Path path(nCity);
    
    if(shape == "circle") {
        // Initialize cities randomly placed on a circumference
        for(int i = 0; i < nCity; ++i) {
            double theta = 2.0 * M_PI * rnd.Rannyu();
            path.set_city(i, City(theta));
        }
    } else if(shape == "square") {
        // Initialize cities randomly placed inside a square
        for(int i = 0; i < nCity; ++i) {
            double x = rnd.Rannyu();
            double y = rnd.Rannyu();
            path.set_city(i, City(x, y));
        }
    } else {
        cerr << "Unknown shape: " << shape << endl;
        return 1;
    }
    
    // Create a directory to save the best paths if it doesn't exist
    string directory = "BEST_PATHS_" + shape;
    fs::create_directory(directory);
    
    // Initialize the population with random paths
    Population pop(rnd, path, PopSize);
    
    // Evolutionary process for nGen generations
    for(int gen = 1; gen <= nGen; ++gen) {
        // Compute fitness and sort the population
        pop.compute_fitness_and_sort();
        
        // Output the best path to a file
        ofstream outBP;
        string filename = directory + "/best_path_gen_" + to_string(gen) + ".txt";
        outBP.open(filename);
        for(const auto& coord : pop.get_best_path()){
            outBP << coord(0) << ", " << coord(1) << endl;
        }
        outBP.close();
        
        // Output the best loss (shortest path length) to a file
        ofstream outBL;
        outBL.open("best_loss_" + shape + ".txt",ios::app);
        outBL << pop.get_best_loss() << endl;
        outBL.close();
        
        // Output the mean loss of the first half of the population to a file
        ofstream outML;
        outML.open("mean_loss_" + shape + ".txt",ios::app);
        outML << pop.get_first_half_mean_loss() << endl;
        outML.close();
        
        // Vector to store offspring
        vector<Individual> offspring;
        offspring.reserve(PopSize);
        
        // Create new population using selection, crossover and mutations
        for (int i = 0; i < PopSize / 2; ++i) {
            Individual mother = pop.select(p); // Select mother
            Individual father = pop.select(p); // Select father
            while (&mother == &father) father = pop.select(p); // Ensure different parents

            Individual child1(rnd, path); // Initialize child1
            Individual child2(rnd, path); // Initialize child2

            // Apply crossover with probability Pcross
            if (rnd.Rannyu() < Pcross) {
                tie(child1, child2) = mother.crossover(father);
            } else {
                // If no crossover, children are copies of parents
                child1 = mother;
                child2 = father;
            }

            // Apply mutations with respective probabilities to both children
            if (rnd.Rannyu() < Pmutate) child1.mutate();
            if (rnd.Rannyu() < Pmutate) child2.mutate();
            if (rnd.Rannyu() < Pshift) child1.shift();
            if (rnd.Rannyu() < Pshift) child2.shift();
            if (rnd.Rannyu() < Ppermutate) child1.permutate();
            if (rnd.Rannyu() < Ppermutate) child2.permutate();
            if (rnd.Rannyu() < Pinvert) child1.invert();
            if (rnd.Rannyu() < Pinvert) child2.invert();

            // Add children to the new population
            offspring.emplace_back(child1);
            offspring.emplace_back(child2);
        }
        
        // Set the new population
        pop.set_population(offspring);
    }
    
    // Save the random number generator seed for reproducibility
    rnd.SaveSeed();

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

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
