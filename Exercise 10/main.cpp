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
#include "mpi.h"

using namespace std;
using namespace arma;
namespace fs = std::filesystem;

void RND_Initialize(Random&, int); // Function to initialize the pseudorandom number generator
 
int main (int argc, char *argv[]){ 
    int size, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    // Initialize the pseudorandom number generator
    Random rnd;
    RND_Initialize(rnd, rank);
    
    // Define problem parameters
    int nCity = 110;          // Number of cities
    int PopSize = 1000;       // Population size
    int nGen = 1000;          // Number of generations
    int nMigr = 20;           // Number of generations between migrations
    double p = 2.0;           // Selection parameter
    double Pcross = 0.70;     // Crossover probability
    double Pmutate = 0.10;    // Mutation probability
    double Pshift = 0.10;     // Shift mutation probability
    double Ppermutate = 0.10; // Permutation probability
    double Pinvert = 0.10;    // Inversion mutation probability
    
    // Create a path with nCity cities
    Path path(nCity);
    
    // Initialize cities read from a file
    ifstream infile("cap_prov_ita.dat");
    if (!infile.is_open()) {
        cerr << "Error opening file: cap_prov_ita.dat" << endl;
        return 1;
    }
    string line;
    int i = 0;
    while (getline(infile, line) && i < nCity) {
        istringstream iss(line);
        double x, y;
        if (iss >> x >> y) {
            path.set_city(i, City(x, y));
            ++i;
        } else {
            cerr << "Error parsing line: " << line << endl;
        }
    }
    infile.close();
    
    // Create a directory to save the best paths if it doesn't exist
    fs::create_directory("BEST_PATHS");
    
    // Initialize the population with random paths
    Population pop(rnd, path, PopSize);
    
    // Evolutionary process for nGen generations
    for(int gen = 1; gen <= nGen; ++gen) {
        // Compute fitness and sort the population
        pop.compute_fitness_and_sort();
        
        if (rank == 0) {
            // Output the best path to a file
            ofstream outBP;
            string filename = "BEST_PATHS/best_path_gen_" + to_string(gen) + ".txt";
            outBP.open(filename);
            for(const auto& coord : pop.get_best_path()){
                outBP << coord(0) << ", " << coord(1) << endl;
            }
            outBP.close();
            
            // Output the best loss (shortest path length) to a file 
            ofstream outBL;
            outBL.open("best_loss.txt",ios::app);
            outBL << pop.get_best_loss() << endl;
            outBL.close();
            
            // Output the mean loss of the first half of the population to a file
            ofstream outML;
            outML.open("mean_loss.txt",ios::app);
            outML << pop.get_first_half_mean_loss() << endl;
            outML.close();
        }
        
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
        
        // Migration step
        if (gen % nMigr == 0) {
            // Find the best individual in the current population
            Individual best_individual = pop.get_best_individual();
            uvec best_genome = best_individual.get_path();
            
            // Buffer for receiving the best individual
            uvec received_genome(nCity + 1);
            
            // Determine the partner for exchange
            int partner = (rank % 2 == 0) ? rank + 1 : rank - 1;
            
            // Perform the exchange using MPI_Sendrecv
            MPI_Sendrecv(
                best_genome.memptr(),     // Pointer to the data to send
                nCity + 1,                // Number of elements to send (number of cities + 1 for a round trip)
                MPI_UNSIGNED_LONG,        // Data type of elements to send
                partner,                  // Rank of the receiving process (partner node)
                0,                        // Message tag for the sent message
                received_genome.memptr(), // Pointer to the buffer to receive data
                nCity + 1,                // Number of elements to receive (number of cities + 1 for a round trip)
                MPI_UNSIGNED_LONG,        // Data type of elements to receive
                partner,                  // Rank of the sending process (partner node)
                0,                        // Message tag for the received message
                MPI_COMM_WORLD,           // MPI communicator
                MPI_STATUS_IGNORE         // Status object (ignore in this case)
            );
            
            // Update the population with the received best individual
            Individual received_individual(received_genome, rnd, path);
            pop.update_population(received_individual);
        }
    }
    
    // Gather all best losses from each node to determine the overall best
    double local_best_loss = pop.get_best_loss();
    double global_best_loss;
    MPI_Reduce(&local_best_loss, &global_best_loss, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);

    // Output the results from the node with the best loss
    if (rank == 0) {
        ofstream out;
        out.open("global_best_loss.txt");
        out << global_best_loss << endl;
        out.close();
        
        ofstream outt;
        outt.open("final_best_path.txt");
        for(const auto& coord : pop.get_best_path()) {
            outt << coord(0) << ", " << coord(1) << endl;
        }
        outt.close();
    }
    
    // Save the random number generator seed for reproducibility
    rnd.SaveSeed();
    
    MPI_Finalize();
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

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
