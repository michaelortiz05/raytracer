.PHONY: build, run, clean

build: program

run: program
	./program $(file)

program: main.cpp
	g++ main.cpp lodepng.cpp image.cpp math.cpp -Wall -Wextra -pedantic -Wno-missing-field-initializers -o program -std=c++20

clean:
	rm -f program
