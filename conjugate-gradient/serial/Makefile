solve_cg:cg.o my_routines.o solve_cg.o
	cc -g -o solve_cg solve_cg.o cg.o my_routines.o

cg.o : cg.c cg.h
	cc -c cg.c

my_routines.o : my_routines.c my_routines.h
	cc -c my_routines.c 

solve_cg.o : solve_cg.c
	cc -c solve_cg.c

clean:
	rm solve_cg solve_cg.o cg.o my_routines.o
