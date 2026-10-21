// One ggml-cpu variant as a Python module: score() is 0 unless this CPU has every ISA extension the variant is built for, and api() is the address of its kernel table.
// setup.py builds this file without the variant's ISA flags, like upstream's <variant>-feats target, so importing the module and calling score() is safe on any x86 CPU.
#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "cpu_kernels.h"
#include "ggml-cpu/arch/x86/cpu-feats.cpp"

#define FT_CAT_(a, b) a##b
#define FT_CAT(a, b) FT_CAT_(a, b)
#define FT_STR_(a) #a
#define FT_STR(a) FT_STR_(a)

namespace {

const ft_ggml_cpu_api kApi = {FT_GGML_CPU_ABI, FT_STR(FT_GGML_VARIANT), ft_ggml_cpu_init, ft_ggml_cpu_row_kernel, ft_ggml_type_name};

PyObject* score(PyObject*, PyObject*) { return PyLong_FromLong(ggml_backend_cpu_x86_score()); }

PyObject* api(PyObject*, PyObject*) { return PyLong_FromVoidPtr(const_cast<ft_ggml_cpu_api*>(&kApi)); }

PyMethodDef kMethods[] = {
    {"score", score, METH_NOARGS, "0 if this CPU lacks an ISA extension the variant is built for, else upstream's variant score."},
    {"api", api, METH_NOARGS, "The address of the variant's kernel table, for _cpu_moe.set_ggml_cpu."},
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef kModule = {PyModuleDef_HEAD_INIT, "_" FT_STR(FT_GGML_VARIANT), nullptr, -1, kMethods};

}  // namespace

PyMODINIT_FUNC FT_CAT(PyInit__, FT_GGML_VARIANT)(void) { return PyModule_Create(&kModule); }
