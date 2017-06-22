# Used specified parameters
KERNELS   = cuda-explicit
CPROFILER = yes
DEBUG     = no
OPTIONS   = -g
CC        = /localhd/mattmar/trunk/obj/bin/clang
CPP       = /localhd/mattmar/trunk/obj/bin/clang++
COMPILER  = CLANG

ABFT_METHOD_CSR_ELEMENT =
ABFT_METHOD_DOUBLE_VECTOR =
INJECT_FAULT = no
NANOS_RECOVERY = no
MB_LOGGING = no
MPI = no
INTERVAL_CHECKS =

ifneq ($(ABFT_METHOD_CSR_ELEMENT),)
  OPTIONS += -DABFT_METHOD_CSR_ELEMENT_$(ABFT_METHOD_CSR_ELEMENT)
endif

ifneq ($(ABFT_METHOD_DOUBLE_VECTOR),)
  OPTIONS += -DABFT_METHOD_DOUBLE_VECTOR_$(ABFT_METHOD_DOUBLE_VECTOR)
endif

ifneq ($(ABFT_METHOD_INT_VECTOR),)
  OPTIONS += -DABFT_METHOD_INT_VECTOR_$(ABFT_METHOD_INT_VECTOR)
endif

ifeq ($(MPI),no)
  OPTIONS += -DNO_MPI
endif

ifeq ($(INJECT_FAULT),yes)
  OPTIONS += -DINJECT_FAULT
endif

ifeq ($(NANOS_RECOVERY),yes)
  OPTIONS += -DNANOS_RECOVERY
endif

ifeq ($(MB_LOGGING),yes)
  OPTIONS += -DMB_LOGGING
endif

ifneq ($(INTERVAL_CHECKS),)
  OPTIONS += -DINTERVAL_CHECKS=$(INTERVAL_CHECKS)
endif

# By default use C compiler and flags, may be overridden by implementation
TL_LINK     = $(CC)
TL_LDFLAGS  = -lrt -lm
TL_COMPILER = $(CC)
TL_FLAGS    = $(CFLAGS)

IS_INTEL = $(shell lscpu | grep GenuineIntel)


include make.flags

# Apply additional options
ifeq ($(CPROFILER), yes)
  OPTIONS += -DENABLE_PROFILING
endif

KERNELS_PATH = c_kernels/$(KERNELS)

vpath %.c drivers/

# Get all control and driver objects
OBJS  = $(patsubst %.c, obj/$(KERNELS)/%.o, $(wildcard *.c))
OBJS += $(patsubst drivers/%.c, obj/$(KERNELS)/%.o, $(wildcard drivers/*.c))

ifneq ($(IS_INTEL),)
  ifneq (,$(filter $(KERNELS),omp3-explicit ompss-explicit))
    # OPTIONS += -DINTEL_ASM
    OBJS += obj/$(KERNELS)/crc_iscsi_v_pcl.o
  endif
endif

# Link together control code, drivers and kernels
tealeaf: make_build_dir build_kernels $(OBJS) Makefile
	$(TL_LINK) $(TL_FLAGS) $(OBJS) $(KERNEL_OBJS) $(TL_LDFLAGS) -o tealeaf.$(KERNELS)
	cp tealeaf.$(KERNELS) tealeaf

include make.deps
include $(KERNELS_PATH)/Makefile

# Rule to make controlling code
obj/$(KERNELS)/%.o: %.c Makefile make.deps $(KERNELS_PATH)/make.deps
	$(TL_COMPILER) $(TL_FLAGS) -I$(KERNELS_PATH) $(OPTIONS) -c $< -o $@

obj/$(KERNELS)/crc_iscsi_v_pcl.o: ABFT/CPU/crc_iscsi_v_pcl.asm
	yasm -f x64 -f elf64 -X gnu -g dwarf2 -D LINUX -o $@ $<

# Create the object file directory and implementation sub-directory
make_build_dir:
	@mkdir -p obj/
	@mkdir -p obj/$(KERNELS)

.PHONY: clean

clean:
	rm -rf obj/ *.o *.a tealeaf mcc_*

## Populates make.deps with dependencies
#obj/%.o: %.cu Makefile make.deps
#	gcc -MM $< >> new.make.deps

