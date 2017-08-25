vpath %.cu $(KERNELS_PATH)

# Override the default compilation options
TL_LINK	  	 = $(CPP)
TL_COMPILER  = $(CPP)
TL_FLAGS 	 = $(CPPFLAGS)
TL_LDFLAGS   += -L$(CUDA_PATH)/lib64 -lcudart $(OMP)

NV_ARCH=KEPLER
CODE_GEN_FERMI=-gencode arch=compute_20,code=sm_21
CODE_GEN_KEPLER=-gencode arch=compute_35,code=sm_35
CODE_GEN_KEPLER_CONSUMER=-gencode arch=compute_30,code=sm_30
CODE_GEN_MAXWELL=-gencode arch=compute_50,code=sm_50
CODE_GEN_PASCAL=-gencode arch=compute_61,code=sm_61

KERNEL_FLAGS   = -I$(CUDA_PATH)/include -restrict -Xcompiler "-O3" \
			 $(CODE_GEN_$(NV_ARCH))

KERNEL_COMPILER = nvcc

ifeq ($(DEBUG),yes)
  KERNEL_FLAGS += -O0 -g -G
else
  KERNEL_FLAGS += -O3  -lineinfo #--ptxas-options="-v"
endif

include $(KERNELS_PATH)/make.deps

KERNEL_SRCS := $(wildcard $(KERNELS_PATH)/*.cu)
KERNEL_OBJS := $(patsubst $(KERNELS_PATH)/%.cu, obj/$(KERNELS)/%.o, $(KERNEL_SRCS))

build_kernels: $(KERNEL_OBJS)
	@echo "Built CUDA kernels"

obj/$(KERNELS)/%.o: %.cu Makefile make.deps
	$(KERNEL_COMPILER) $(KERNEL_FLAGS) -I$(KERNELS_PATH) $(OPTIONS) -c $< -o $@

