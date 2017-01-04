vpath %.c $(KERNELS_PATH)

include $(KERNELS_PATH)/make.deps

TL_LDFLAGS += --ompss --debug -k

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.c)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.c, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built OmpSs (explicit) kernels $OMP"

