CC=					OMPI_CC=gcc mpicc
CFLAGS=				-O3 -Wall -ggdb3 -fopenmp
LD=
DEFS=

ifeq ($(CC), mpicc)
	DEFS+=			-D_MPI
endif

all: cdb

cdb: cdb.c
	$(CC) $(CFLAGS) $(DEFS) -o $@ $^ $(LD)

clear:
	rm -rf *.cdb *.cdb.*

clean: clear
	rm -rf *.o cdb cdb.dSYM