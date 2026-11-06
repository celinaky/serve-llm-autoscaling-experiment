"""Process-local overlay: leave installed symlinks and original checkouts intact."""
import importlib.abc
import importlib.util
import sys
from pathlib import Path

class OnlinePrefillFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in ('ray.serve', 'ray.llm'):
            directory = Path('/mnt/local_storage/ray-online-prefill/python').joinpath(*fullname.split('.'))
            return importlib.util.spec_from_file_location(fullname, directory / '__init__.py', submodule_search_locations=[str(directory), str(Path("/mnt/local_storage/ray/python").joinpath(*fullname.split(".")))])
        return None
sys.meta_path.insert(0, OnlinePrefillFinder())
