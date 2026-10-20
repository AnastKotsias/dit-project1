BUILD_DIR ?= build
JOBS ?= 2

.PHONY: all configure clean

all: configure
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

configure:
	cmake -S . -B $(BUILD_DIR)

clean:
	cmake --build $(BUILD_DIR) --target clean
