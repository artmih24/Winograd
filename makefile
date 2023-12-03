ifeq ($(OS),Windows_NT) 
RM = del /Q /F
CP = copy /Y
EXE = a.exe
ifdef ComSpec
SHELL := $(ComSpec)
endif
ifdef COMSPEC
SHELL := $(COMSPEC)
endif
else
RM = rm -rf
CP = cp -f
EXE = a.out
endif

all: main.o gemm.o
	g++ main.o gemm.o -o $(EXE)

main.o:
	g++ -c Winograd/main.cpp

gemm.o:
	g++ -c Winograd/gemm.cpp

clean:
	$(RM) *.o $(EXE)