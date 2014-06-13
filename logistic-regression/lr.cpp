#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <set>
#include <map>

#include <stdio.h>
#include <stdlib.h>

#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

// refer to matrix row
#include <boost/numeric/ublas/matrix_proxy.hpp>

#include "util.hpp"
#include "data_loader.hpp"
#include "gradient_descent.hpp"


using namespace std;
using namespace boost::numeric::ublas;


bool debug = true;

// target: max { sum {log f(y(i)z(i)}} for i in (1, n) where f(x) = 1/1+e**(-x)
// and z(i) = sum(w(k) * x(i)(k)) for k in (1, l) where i denotes the ith training instance
// and k denotes the kth feature. 
// The gradient of the log-likehood with respect to the kth weight is:
// gra = sum{y(i)x(i)(k)f(-y(i)z(i))}, then we know how to update the weight in each iteration:
// w(k)(t+1) = w(k)(t) + e * gra
void lr_without_regularization(boost::numeric::ublas::matrix<double>& x,
        boost::numeric::ublas::vector<double>& y
        ) {

    // the convergence rate
    double epsilon = 0.0001;
    // the learning rate
    double gamma = 0.00005;
    int max_iters = 2000;

    boost::numeric::ublas::vector<double> w = gradient_descent(x, y, epsilon, gamma, max_iters);     

    cout << "The optimal weight is: " << w << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " data_file" << endl;
        return -1;
    }

    const int record_num = 270;
    const int dim_num = 13 + 1;

    boost::numeric::ublas::vector<double> y(record_num);
    boost::numeric::ublas::matrix<double> x(record_num, dim_num);
    SimpleDataLoader loader(record_num, dim_num);
    loader.load_file(argv[1], y, x);

    // lr_method
    lr_without_regularization(x, y);

    return 0;
}
