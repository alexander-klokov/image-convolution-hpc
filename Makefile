.PHONY: build run_bl see_bl

# executables
BUILD_DIR = build
EXE_BL = exe_baseline

# input/output
INPUT=input/pebble.pgm
OUTPUT_BL=output/pebble_blurred_baseline.pgm

build:
	@mkdir -p $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake ..
	@cd $(BUILD_DIR) && cmake --build .

# baseline
run_bl:
	$(BUILD_DIR)/$(EXE_BL) ${INPUT} ${OUTPUT_BL}

see_bl:
	gimp ${OUTPUT_BL}

# clean up
clean:
	@rm -rf $(BUILD_DIR) output profile