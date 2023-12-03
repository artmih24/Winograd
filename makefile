all: main.o gemm.o
	g++ -std=c++11 main.o gemm.o -o Winograd.exe

main.o:
	g++ -std=c++11 -c Winograd/main.cpp -o main.o

gemm.o:
	g++ -std=c++11 -c Winograd/gemm.cpp -o gemm.o