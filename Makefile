.PHONY: build, run, clean

build: program

run: program
	./program $(file)

program: main.cpp
	g++-13 main.cpp lodepng.cpp image.cpp math.cpp -Wall -Wextra -pedantic -Wno-missing-field-initializers -o program -std=c++20 -fopenmp

clean:
	rm -f program
