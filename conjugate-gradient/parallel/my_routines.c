#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include "cblas.h"

#ifndef min
#define min(a, b) ((a) < (b) ? (a) : (b))
#endif

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif

/* Με μονοδιαστατο πίνακα!!!!
* Πολλαπλασιασμός πίνακα επί διάνυσμα - 
 */
void matvec_blas(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE TransA, const int M, const int N,
                 const double alpha, const double  *A, const int lda,
                 const double  *X, const int incX, const double beta,
                 double  *Y, const int incY)
{
    int i, j;		
    
    for (i = 0; i < N; ++i) {
        Y[i] = 0;
        for (j = 0; j < N; ++j) {
            Y[i] += A[i*N+j]*X[j];
        }
    }
}

/*
void matvec_blas(double *Ax, double *Adata, double *xvect, int n)
{
    int i, j;		
    
    for (i = 0; i < n; ++i) {
        Ax[i] = 0;
        for (j = 0; j < n; ++j) {
            Ax[i] += Adata[i*n+j]*xvect[j];
        }
    }
}*/

/* Με μονοδιαστατο πίνακα!!!!
 * Δημιουργία ενός Jacobi preconditioner - Τα μηδενικά δεν αποθηκεύονται 
 * Mij = Aij, i=j
 * Mij = 0, i<>j
 */
void jacobi_precond_blas(double *M, double *Adata, int n)
{
    int i;		
    
    for (i = 0; i < n; i++) {
           M[i] += Adata[i*n+i];
    }
}

/* Εφαρμογή του Jacobi preconditioner
 * 
 */
void psolve(double *Minvx, double *Mdata, double *x, int n)
{
    int i;		
    
    for(i=0; i<n; i++){
	Minvx[i] = 1/Mdata[i]*x[i];
    }
}

/* Παραγωγή ενός τυχαίου διανύσματος vec μεγέθους n - solution vector
 * 
 */
void generate_randvec(double *vec, int n)
{
    int i;	

    for(i=0; i<n; i++){
	vec[i] = (double)random()/(RAND_MAX+1.0);
    }
	
}

//Γέμισμα τυχαίου πίνακα από αρχείο
void fillMat(double *mat, int n, FILE* f){	
int i;
	for(i=0; i<n*n; i++)
		fscanf(f, "%lf", &mat[i]);
}

void generate_randmat_blas(double *mat, int n)
{
    int i, j, k;	
    double *randmat;
    double *mattrans;	

    randmat = malloc(n *n* sizeof(double));
 		if( randmat == NULL)  	    printf("randmat--Out of memory");  
	  	  
	
     mattrans = malloc(n *n* sizeof(double));
 		if( mattrans == NULL)  	    printf("mattrans--Out of memory");  
	  	  

    for(i=0; i<n; i++){
	for(j=0; j<n; j++){
		randmat[i*n+j] = (double)random()/(RAND_MAX+1.0);
    	}
    }

    for(i=0; i<n; i++){
	for(j=0; j<n; j++){
		mattrans[i*n+j] = randmat[j*n+i];
    	}
    }

    for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			for(k=0;k<n;k++){
				mat[i*n+k] += mattrans[i*n+j]*randmat[j*n+k];
			}
		}
    }

    //free memory
    free(randmat);
    free(mattrans);
	
}
/* Παραγωγή ενός τυχαίου συστήματος A*x=b μεγέθους n
 * 
 */
void generate_randsys_blas(double *A, double *x, double *b, int n, FILE* f)
{
    int i, j;

   // generate_randmat_blas(A, n);
    fillMat(A, n, f);
    generate_randvec(x, n);	

    matvec_blas(CblasRowMajor, CblasNoTrans, n, n, 1.0, A, n, x, 1, 0.0, b, 1);
}

/*int main(int argc, char **argv)
{
	int i, j;	
	double **Adata;	//Ο πίνακας του συστήματος
	double *x;	//Η λύση του συστήματος (που παράγεται για το τυχαίο σύστημα)
	double *cg_x;	//Η λύση του συστήματος που προκύπτει από τη cg
	double *b;	//Το δεξί μέλος του συστήματος
	int n = atoi(argv[1]);	//Το μέγεθος του συστήματος

	//Memory allocation
	Adata = calloc(n, sizeof(double*));
 		if( Adata == NULL)  	    printf("Adata--Out of memory");  
	  	  
		Adata[0] = calloc(n * n, sizeof(double));
		for(i = 1; i < n; i++)
			Adata[i] = Adata[0] + i * n;

	x = calloc(n, sizeof(double));
	cg_x = calloc(n, sizeof(double));
	b = calloc(n, sizeof(double));

	//Παραγωγή τυχαίου συστήματος
	generate_randsys(Adata, x, b, n);

	for(i=0; i<n; i++){
		for(j=0; j<n; j++){
			printf("Adata[%d][%d] = %lf\t", i, j, Adata[i][j]);  
		}
		printf("\n");
	}
}*/

