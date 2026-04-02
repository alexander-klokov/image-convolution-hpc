# The Magic Trick for Positional Arguments
ifeq (run,$(firstword $(MAKECMDGOALS)))
  # filter-out removes 'run' from the goals, leaving only the ID
  RUN_ARGS := $(filter-out run,$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

# Default to 0
KERNEL_ID = $(strip $(if $(RUN_ARGS),$(RUN_ARGS),0))

# ------------------------------------------------

.PHONY: build run see clean

# executables
BUILD_DIR = build
EXEC = convolution_benchmark

# input/output
INPUT = input/pebble.pgm
# Dynamic output based on the kernel ID
OUTPUT = output/pebble_blurred_$(KERNEL_ID).pgm

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake -DCMAKE_BUILD_TYPE=Release ..
	@cd $(BUILD_DIR) && cmake --build . -j 12

# run target with dynamic kernel injection
run: build
	@mkdir -p output
	@echo "========================================"
	@echo "Executing Kernel ID: $(KERNEL_ID)"
	@echo "========================================"
	./$(BUILD_DIR)/$(EXEC) $(INPUT) $(OUTPUT) $(KERNEL_ID)

# Open the dynamically generated image
see:
	gimp $(OUTPUT)

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile