.PHONY: all clean

# the ./search link lets the program run as ./search
all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	cmake --build build --parallel
	ln -sf build/search search

clean:
	rm -rf build search
