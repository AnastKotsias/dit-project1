.PHONY: all test clean

# the ./search and ./verify links let the programs run from the repository root
all:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_SEARCH_APP=ON
	cmake --build build --parallel
	ln -sf build/search search
	ln -sf build/verify verify

test:
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DBUILD_SEARCH_APP=OFF
	cmake --build build --target image_search_tests --parallel
	ctest --test-dir build --output-on-failure

clean:
	rm -rf build search verify
