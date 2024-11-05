import os
from torch.utils.cpp_extension import load


def load_src(name, verbose=False):
    module_path = os.path.dirname(__file__)
    print(f"--->>> Compiling {name}")
    return load(name,
                sources=[os.path.join(module_path, f"op/{name}.cpp"),
                         os.path.join(module_path, f"op/{name}_kernel.cu")],
                extra_include_paths=[os.path.join(module_path, "include")],
                verbose=verbose)
