vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

TL_LDFLAGS += $(OMP) -L/nfs/home/george/builds/fti-$(OMPI_CC)/lib -lfti

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	LD_LIBRARY_PATH=~/builds/fti-$(OMPI_CC)/lib:$LD_LIBRARY_PATH
	@echo "Built OpenMP 3.0 (explicit) kernels"

