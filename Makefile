# The Magic Trick for Positional Arguments
ifneq (,$(filter run run8k profile profile8k see,$(firstword $(MAKECMDGOALS))))
  # leaving only the ID
  RUN_ARGS := $(filter-out run profile see,$(MAKECMDGOALS))
  $(eval $(RUN_ARGS):;@:)
endif

# Default to 0
KERNEL_ID = $(strip $(if $(RUN_ARGS),$(RUN_ARGS),0))

# ------------------------------------------------

.PHONY: build run see clean profile

# executables
BUILD_DIR := build
EXEC := convolution_benchmark
PERF := /usr/lib/linux-tools/6.8.0-110-generic/perf

# input/output
INPUT = input/pebble.pgm
INPUT8K = input/pebble_8k.pgm
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

# profile the target kernel
profile:
	$(PERF) stat -d ./$(BUILD_DIR)/$(EXEC) $(INPUT) $(OUTPUT) $(KERNEL_ID)

# run for the 8k 43MB input
run8k: build
	@mkdir -p output
	@echo "========================================"
	@echo "Executing Kernel ID: $(KERNEL_ID)"
	@echo "========================================"
	./$(BUILD_DIR)/$(EXEC) $(INPUT_8K) $(OUTPUT) $(KERNEL_ID)

# profile for the 8k 43MB input
profile8k:
	$(PERF) stat -d ./$(BUILD_DIR)/$(EXEC) $(INPUT8K) $(OUTPUT) $(KERNEL_ID)

# open the blurred image
see:
	gimp $(OUTPUT)

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile