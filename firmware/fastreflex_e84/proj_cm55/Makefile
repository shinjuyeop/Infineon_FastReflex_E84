################################################################################
# FastReflex E84 CM55 + Ethos-U55 non-release HIL runtime
################################################################################

include ../common.mk

APPNAME=proj_cm55
VERBOSE=

CORE=CM55
CORE_NAME=CM55_0
NPU=U55

# Generated model conversions stay outside tracked source and are staged by
# tools/deployment.py after target-specific Vela compilation.
NN_MODEL_FOLDER=mtb_ml_gen
CY_IGNORE+=$(NN_MODEL_FOLDER)
SOURCES+=$(wildcard $(NN_MODEL_FOLDER)/mtb_ml_models/*.c)
SOURCES+=../shared_src/app_common.c
INCLUDES+=$(NN_MODEL_FOLDER)/mtb_ml_models ../shared_src/

COMPONENTS+=U55 ML_TFLM ML_INT8x8
DEFINES+=TF_LITE_STATIC_MEMORY CY_ML_MODEL_MEM=.cy_socmem_data
# The official middleware's conditional policy performs one whole-cache clean
# and invalidate per invocation instead of repeating maintenance per NPU region.
DEFINES+=MTB_ML_ETHOSU_CACHE_MGMT_TYPE=MTB_ML_ETHOSU_CACHE_MGMT_CONDITIONAL
DEFINES+=ARM_MATH_HELIUM ARM_MATH_DSP ARM_MATH_AUTOVECTORIZE

VFP_SELECT+=
CFLAGS+=
CXXFLAGS+=
ASFLAGS+=
LDFLAGS+=
LDLIBS+=
LINKER_SCRIPT=
PREBUILD=
POSTBUILD=

CY_APP_PATH=
CY_GETLIBS_SHARED_PATH=../..
CY_GETLIBS_SHARED_NAME=mtb_shared

include $(CY_TOOLS_DIR)/make/start.mk
