CC = mpicxx
CFLAGS = -O3 --std=c++17
AFLAGS = -larmadillo
all: main.exe
main.exe : main.o TSP.o random.o
	$(CC) main.o TSP.o random.o -o main.exe $(AFLAGS)
main.o : main.cpp TSP.h random.h
	$(CC) -c main.cpp -o main.o $(CFLAGS)
TSP.o : TSP.cpp TSP.h random.h
	$(CC) -c TSP.cpp -o TSP.o $(CFLAGS)
random.o : random.cpp random.h
	$(CC) -c random.cpp -o random.o $(CFLAGS)
clean :
	rm *.o main.exe seed.out
