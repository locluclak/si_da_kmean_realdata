from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "solveinequalities.interval",   # 👈 put it inside the package
        ["solveinequalities/interval.cpp"],  # 👈 move cpp file inside the package
        include_dirs=[pybind11.get_include()],
        language="c++"
    )
]

setup(
    name="solveinequalities",
    packages=["solveinequalities"],  # 👈 declare the package
    ext_modules=ext_modules,
)
