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
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <utility>
#include "random.h"
#include "TSP.h"

using namespace std;

//------------------------------------------//
//--------------- Class City ---------------//
//------------------------------------------//

City::City() : _coord({0.0, 0.0}) {}

City::City(double x, double y) : _coord({x, y}) {}

City::City(double theta) : _coord({cos(theta), sin(theta)}) {}

City::~City() {}

void City::set_coord(double x, double y) { _coord = {x, y}; }

void City::set_coord_theta(double theta) { _coord = {cos(theta), sin(theta)}; }

vec2 City::get_coord() const { return _coord; }

void City::print_coord() const {
    cout << "Coordinate: (" << _coord(0) << ", " << _coord(1) << ")" << endl;
}

double City::dist(const City& otherCity) const {
    return norm(_coord - otherCity.get_coord());
}

//-----------------------------------------//
//--------------- Class Path --------------//
//-----------------------------------------//

Path::Path(int N) : _State(N) {}

Path::~Path() {}

void Path::set_city(int index, const City& newCity) { _State(index) = newCity; }

City Path::get_city(int index) const { return _State(index); }

void Path::print_city() const {
    for (const auto& city : _State) {
        city.print_coord();
    }
}

int Path::get_ncity() const { return _State.n_elem; }

double Path::loss_function(const uvec& seq) const {
    double loss = 0.0;
    for (size_t i = 0; i < seq.n_elem - 1; ++i) {
        loss += _State(seq(i)).dist(_State(seq(i + 1)));
    }
    return loss;
}

//-----------------------------------------//
//------------ Class Individual -----------//
//-----------------------------------------//

Individual::Individual(Random& rnd, const Path& path) : _sequence(path.get_ncity() + 1), _rnd(rnd), _path(path) {
    iota(_sequence.begin(), _sequence.end() - 1, 0);
    _sequence.back() = 0;
    for (int i = 0; i < _path.get_ncity(); ++i) mutate();
}

Individual::Individual(const uvec& seq, Random& rnd, const Path& path) : _sequence(seq), _rnd(rnd), _path(path) {}

Individual::Individual(const Individual& other)
    : _sequence(other._sequence), _rnd(other._rnd), _path(other._path) {}

Individual& Individual::operator=(const Individual& other) {
    if (this != &other) {
        _sequence = other._sequence;
        _rnd = other._rnd;
    }
    return *this;
}

Individual::~Individual() {}

void Individual::set_path(const uvec& seq) { _sequence = seq; }

uvec Individual::get_path() const { return _sequence; }

double Individual::get_loss() const { return _path.loss_function(_sequence); }

void Individual::print_path() const {
    cout << "Path: ";
    for (const auto& s : _sequence) {
        cout << s << " ";
    }
    cout << endl;
}

bool Individual::check_bonds() const {
    if (_sequence.front() != 0 || _sequence.back() != 0) return false;
    unordered_set<int> idx_set(_sequence.begin(), _sequence.end() - 1);
    return idx_set.size() == _sequence.size() - 1;
}

void Individual::mutate() {
    if (_sequence.n_elem > 1) {
        int idx1 = _rnd.Rannyu(1, _sequence.n_elem - 1);
        int idx2 = _rnd.Rannyu(1, _sequence.n_elem - 1);
        while (idx1 == idx2) {
            idx2 = _rnd.Rannyu(1, _sequence.n_elem - 1);
        }
        swap(_sequence(idx1), _sequence(idx2));
    }
}

void Individual::shift() {
    int m = _rnd.Rannyu(1, _sequence.n_elem - 2);
    int n = _rnd.Rannyu(1, _sequence.n_elem - m - 1);
    int start = _rnd.Rannyu(1, _sequence.n_elem - m - n);
    uvec temp = _sequence.subvec(start, start + m - 1);
    _sequence.shed_rows(start, start + m - 1);
    _sequence.insert_rows(start + n, temp);
}

void Individual::permutate() {
    int m = _rnd.Rannyu(2, _sequence.n_elem / 2);
    int start1 = _rnd.Rannyu(1, _sequence.n_elem - 2 * m);
    int start2 = _rnd.Rannyu(1, _sequence.n_elem - m);
    while (abs(start1 - start2) < m) {
        start2 = _rnd.Rannyu(1, _sequence.n_elem - m);
    }
    for (int i = 0; i < m; ++i) {
        swap(_sequence(start1 + i), _sequence(start2 + i));
    }
}

void Individual::invert() {
    int m = _rnd.Rannyu(2, _sequence.n_elem - 1);
    int start = _rnd.Rannyu(1, _sequence.n_elem - m - 1);
    reverse(_sequence.begin() + start, _sequence.begin() + start + m);
}

pair<Individual, Individual> Individual::crossover(const Individual& partner) {
    int cut_point = _rnd.Rannyu(1, _sequence.n_elem - 2);
    uvec sequence1 = _sequence;
    uvec sequence2 = partner._sequence;
    unordered_set<int> cities_in_child1(sequence1.begin() + 1, sequence1.begin() + cut_point + 1);
    unordered_set<int> cities_in_child2(sequence2.begin() + 1, sequence2.begin() + cut_point + 1);
    int index1 = cut_point + 1;
    int index2 = cut_point + 1;
    
    for (size_t i = 1; i < _sequence.n_elem - 1; ++i) {
        if (cities_in_child1.find(partner._sequence(i)) == cities_in_child1.end()) {
            sequence1(index1++) = partner._sequence(i);
            cities_in_child1.insert(partner._sequence(i));
        }
        if (cities_in_child2.find(_sequence(i)) == cities_in_child2.end()) {
            sequence2(index2++) = _sequence(i);
            cities_in_child2.insert(_sequence(i));
        }
    }
    
    return {Individual(sequence1, _rnd, _path), Individual(sequence2, _rnd, _path)};
}

//-----------------------------------------//
//------------ Class Population -----------//
//-----------------------------------------//

Population::Population(Random& rnd, const Path& path, int size) : _rnd(rnd), _path(path) {
    for (int i = 0; i < size; ++i) {
        _individuals.emplace_back(_rnd, _path);
    }
}

Population::Population(const vector<Individual>& individuals, Random& rnd, const Path& path) : _individuals(individuals), _rnd(rnd), _path(path) {}

Population::Population(const Population& other)
    : _individuals(other._individuals), _rnd(other._rnd), _path(other._path) {}

Population& Population::operator=(const Population& other) {
    if (this != &other) {
        _individuals = other._individuals;
        _rnd = other._rnd;
    }
    return *this;
}

Population::~Population() {}

void Population::set_population(const vector<Individual>& individuals) {
    _individuals = individuals;
}

void Population::add_individual(const Individual& individual) {
    _individuals.push_back(individual);
}

Individual Population::get_individual(int idx) const { return _individuals[idx]; }

int Population::get_population_size() const { return _individuals.size(); }

void Population::print_population() const {
    for (size_t i = 0; i < _individuals.size(); ++i) {
        cout << "Individual " << i + 1 << ": ";
        _individuals[i].print_path();
    }
}

bool Population::check_population() const {
    return all_of(_individuals.begin(), _individuals.end(),
                  [](const Individual& ind) { return ind.check_bonds(); });
}

void Population::compute_fitness_and_sort() {
    sort(_individuals.begin(), _individuals.end(),
        [](const Individual& a, const Individual& b) {
            return a.get_loss() < b.get_loss();
        });
}

Individual Population::get_best_individual() const {
    auto best_it = min_element(_individuals.begin(), _individuals.end(),
                               [](const Individual& a, const Individual& b) {
                                    return a.get_loss() < b.get_loss();
                                });
    return *best_it;
}

Individual Population::select(double p) const {
    int j = _individuals.size() * pow(_rnd.Rannyu(), p);
    return _individuals[j];
}

double Population::get_best_loss() const {
    const Individual& best_individual = this->get_best_individual();
    return best_individual.get_loss();
}

vector<vec2> Population::get_best_path() const {
    const Individual& best_individual = this->get_best_individual();
    vector<vec2> coord_best_path;
    for (const auto& idx : best_individual.get_path()) {
        coord_best_path.push_back(_path.get_city(idx).get_coord());
    }
    return coord_best_path;
}

double Population::get_first_half_mean_loss() {
    this->compute_fitness_and_sort();
    size_t half_size = _individuals.size() / 2;
    double total_distance = 0.0;
    for (size_t i = 0; i < half_size; ++i) {
        total_distance += _individuals[i].get_loss();
    }
    return total_distance / half_size;
}

void Population::update_population(const Individual& received_individual) {
    auto worst_it = max_element(_individuals.begin(), _individuals.end(),
                                [](const Individual& a, const Individual& b) {
                                    return a.get_loss() < b.get_loss();
                                });
    if (worst_it != _individuals.end()) *worst_it = received_individual;
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
