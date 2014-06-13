#include <iostream>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/numeric/ublas/matrix_proxy.hpp>
#include <stdio.h>
#include <stdlib.h>

using namespace std;
using namespace boost::numeric::ublas;


boost::numeric::ublas::vector<double> gradient_descent(boost::numeric::ublas::matrix<double>& x, boost::numeric::ublas::vector<double>& y, double epsilon, double gamma, int max_iters) {

	int iter = 1;
	
	boost::numeric::ublas::vector<double> weight_old(x.size2());
	for (size_t i = 0; i < weight_old.size(); i++) 
		weight_old(i) = 0;
	
	cout << "old weight: " << weight_old << endl;
	
	boost::numeric::ublas::vector<double> weight_new(x.size2());
	for (size_t i = 0; i < weight_new.size(); i++) 
		weight_new(i) = 0;
	
	cout << "new weight: " << weight_new << endl;

	
	while(true) {
		for (size_t k = 0; k < weight_new.size(); k++) {
			double gradient = 0;
			for (size_t i = 0; i < x.size1(); i++) {
				double z_i = 0;
				for (size_t j = 0; j < weight_old.size(); j++) {
					// w^T * x
					z_i += weight_old(j) * x(i, j);
				}

				gradient = y(i) * x(i, k) * sigmoid(-y(i) * z_i);
			}
			
			weight_new(k) = weight_old(k) + gamma * gradient;	
		}

		double dist = norm(weight_new, weight_old);

		if (dist < epsilon){ 
			cout << "The best weight is:" << weight_new << endl;
			cout << "After " << iter << " th iteration" << endl;
			break;
		}
		else
			weight_old.swap(weight_new);

		iter += 1;

		if (iter >= max_iters) {
			cout << "The best weight is:" << weight_new << endl;
			cout << "After " << iter << " th iteration" << endl;

			break;
		}

		cout << "===========================================" << endl;
		cout << "The " << iter << " th iteration, weight:" << endl;
		cout << weight_new << endl << endl;
		cout << "The best weight:" << endl;
		cout << "ending" << endl;
	}	

	return weight_new;
}
