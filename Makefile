.PHONY: build, run, clean

build: program

run: program
	./program $(file)

program: main.cpp
	nvcc main.cpp lodepng.cpp image.cu math.cu -o program 

clean:
	rm -f program


