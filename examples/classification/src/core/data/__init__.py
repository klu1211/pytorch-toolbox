from pathlib import Path
from pytorch_toolbox.utils.pipeline import create_lookup_with_relative_modules_from_path


lookup = create_lookup_with_relative_modules_from_path(Path(__file__).parent)
print(lookup)

# x = list(submodule_paths)
# mod = load_module_from_path(x[0])
# lookup = create_lookup_from_module(mod)
# functions = load_pointers_from_modules(mod)
# classes = load_classes_from_modules(mod)
# print("ROOR")
