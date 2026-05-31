CC = gcc
CFLAGS = -O2 -fopenmp
SVE_CFLAGS ?= -O3 -fopenmp -march=armv8-a+sve

FC = gfortran
FFLAGS = -O2 -fopenmp

all: stream_f.exe stream_c.exe stream_all.exe

stream_f.exe: stream.f mysecond.o
	$(CC) $(CFLAGS) -c mysecond.c
	$(FC) $(FFLAGS) -c stream.f
	$(FC) $(FFLAGS) stream.o mysecond.o -o stream_f.exe

stream_c.exe: stream.c
	$(CC) $(CFLAGS) stream.c -o stream_c.exe

stream_all.exe: stream_all.c
	$(CC) $(CFLAGS) stream_all.c -o stream_all.exe

stream_sve.exe: stream_sve.c
	$(CC) $(SVE_CFLAGS) stream_sve.c -o stream_sve.exe

clean:
	rm -f stream_f.exe stream_c.exe stream_all.exe stream_sve.exe *.o

# an example of a more complex build line for the Intel icc compiler
stream.icc: stream.c
	icc -O3 -xCORE-AVX2 -ffreestanding -qopenmp -DSTREAM_ARRAY_SIZE=80000000 -DNTIMES=20 stream.c -o stream.omp.AVX2.80M.20x.icc
