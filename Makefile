BUILD_DIR ?= build
# passed explicitly, so a build directory cached as Debug cannot silently produce a slow ./search
BUILD_TYPE ?= Release
JOBS ?= 2

.PHONY: all configure clean

# builds with CMake, then links ./search to the executable so the assignment's ./search command works
all: configure
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
	ln -sf $(BUILD_DIR)/search search

configure:
	cmake -S . -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=$(BUILD_TYPE)

# only asks CMake to clean a build directory that has been configured
clean:
	if [ -f $(BUILD_DIR)/CMakeCache.txt ]; then cmake --build $(BUILD_DIR) --target clean; fi
	rm -f search
