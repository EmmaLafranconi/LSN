/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/

#ifndef __TSP__
#define __TSP__

#include <vector>
#include <armadillo>
#include "random.h"

using namespace std;
using namespace arma;

// Class representing a city in the TSP problem
class City {

private:
    vec2 _coord;  // Coordinates of the city (2D vector)

public:
    City();  // Default constructor
    City(double x, double y);  // Constructor with x, y coordinates
    City(double theta);  // Constructor with polar coordinates (angle)
    ~City();  // Destructor
    void set_coord(double x, double y);  // Set coordinates using x, y
    void set_coord_theta(double theta);  // Set coordinates using angle (theta)
    vec2 get_coord() const;  // Get coordinates of the city
    void print_coord() const;  // Print coordinates of the city
    double dist(const City& otherCity) const;  // Calculate distance to another city
};

// Class representing a path (sequence of cities) in the TSP problem
class Path {

private:
    field<City> _State;  // Field (array) of cities in the path

public:
    Path(int N);  // Constructor with number of cities
    ~Path();  // Destructor
    void set_city(int index, const City& newCity);  // Set city at specific index
    City get_city(int index) const;  // Get city at specific index
    void print_city() const;  // Print all cities in the path
    int get_ncity() const;  // Get number of cities in the path
    double loss_function(const uvec& seq) const;  // Calculate the loss (total distance) for a given sequence of cities
};

// Class representing an individual (solution) in the population
class Individual {

private:
    uvec _sequence;  // Sequence of city indices representing the path
    Random& _rnd;  // Reference to random number generator
    const Path& _path;  // Reference to the path

public:
    Individual(Random& rnd, const Path& path);  // Constructor with random number generator and path
    Individual(const uvec& seq, Random& rnd, const Path& path);  // Constructor with initial sequence, random number generator and path
    Individual(const Individual& other);  // Copy constructor
    Individual& operator=(const Individual& other); // Assignment operator
    ~Individual();  // Destructor
    void set_path(const uvec& seq);  // Set path sequence
    uvec get_path() const;  // Get path sequence
    double get_loss() const;  // Get loss (total distance) of the path
    void print_path() const;  // Print path sequence
    bool check_bonds() const;  // Check validity of the path (no duplicate cities, start and end in hometown)
    void mutate();  // Apply pair permutation of cities to the path
    void shift();  // Apply shift of +n positions for m contiguous cities to the path
    void permutate();  // Apply permutation among m contiguous cities with other m contiguous cities to the path
    void invert();  // Apply inversion of the order of m cities to the path
    pair<Individual, Individual> crossover(const Individual& partner);  // Perform crossover with another individual
};

// Class representing a population of individuals (solutions)
class Population {

private:
    vector<Individual> _individuals;  // Vector of individuals
    Random& _rnd;  // Reference to random number generator
    const Path& _path;  // Reference to the path

public:
    Population(Random& rnd, const Path& path, int size);  // Constructor with random number generator, path, and population size
    Population(const vector<Individual>& individuals, Random& rnd, const Path& path);  // Constructor with initial individuals, random number generator and path
    Population(const Population& other);  // Copy constructor
    Population& operator=(const Population& other); // Assignment operator
    ~Population();  // Destructor
    void set_population(const vector<Individual>& individuals);  // Set population sequences
    void add_individual(const Individual& individual);  // Add an individual to the population
    Individual get_individual(int idx) const;  // Get an individual at specific index
    int get_population_size() const;  // Get size of the population
    void print_population() const;  // Print all individuals in the population
    bool check_population() const;  // Check validity of the population (no duplicate cities in the paths, start and end in hometown)
    void compute_fitness_and_sort();  // Compute fitness of all individuals and sort them
    Individual get_best_individual() const;  // Get the best individual in the population
    Individual select(double p) const;  // Select an individual based on probability
    double get_best_loss() const;  // Get the best (lowest) loss in the population
    vector<vec2> get_best_path() const;  // Get the best path (sequence of cities)
    double get_first_half_mean_loss();  // Get mean loss of the first half of the population
    void update_population(const Individual& received_individual);  // Update the population replacing the worst individual with the received one
};

#endif // __TSP__

/****************************************************************
*****************************************************************
    _/    _/  _/_/_/  _/       Numerical Simulation Laboratory
   _/_/  _/ _/       _/       Physics Department
  _/  _/_/    _/    _/       Universita' degli Studi di Milano
 _/    _/       _/ _/       Emma Lafranconi
_/    _/  _/_/_/  _/_/_/_/ email: emma.lafranconi@studenti.unimi.it
*****************************************************************
*****************************************************************/
