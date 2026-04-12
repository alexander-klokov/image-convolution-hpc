# The Magic Trick for Positional Arguments
ifneq (,$(filter run runmpi see,$(firstword $(MAKECMDGOALS))))
  # filter-out removes 'run' and 'runmpi' from the goals, leaving only the ID
  RUN_ARGS := $(filter-out run runmpi see,$(MAKECMDGOALS))
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
# dynamic output based on the kernel ID
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

runmpi: build
	@mkdir -p output
	@echo "========================================"
	@echo "Executing Kernel ID: $(KERNEL_ID)"
	@echo "========================================"
	mpirun -np 6 --bind-to core ./$(BUILD_DIR)/$(EXEC_MPI) $(INPUT) $(OUTPUT)

# open the blurred image
see:
	gimp $(OUTPUT)

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile