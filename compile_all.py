import os
import shutil
import subprocess

from setuptools import Extension
from Cython.Build import cythonize
import numpy

# Folder where compiled files will be saved
OUTPUT_FOLDER = "compiled_files"

# Ensure the output folder exists
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# Get all .py files in the current directory except this script
py_files = [f for f in os.listdir() if f.endswith(".py") and f != "compile_all.py"]

for py_file in py_files:
    pyx_file = py_file.replace(".py", ".pyx")
    shutil.copy(py_file, pyx_file)  # Copy .py file to .pyx

    # Create a temporary setup.py for compilation
    setup_code = f"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

setup(
    ext_modules=cythonize(
        Extension(
            "{pyx_file.replace('.pyx', '')}",
            ["{pyx_file}"],
            include_dirs=[numpy.get_include()]
        ),
        compiler_directives={{"language_level": "3"}}
    )
)
"""
    with open("temp_setup.py", "w") as f:
        f.write(setup_code)

    # Run the setup script
    subprocess.run(["python", "temp_setup.py", "build_ext", "--inplace"], check=True)

    # Move compiled .pyd file to the output folder
    compiled_file = f"{pyx_file.replace('.pyx', '')}.cp{os.sys.version_info.major}{os.sys.version_info.minor}-win_amd64.pyd"
    if os.path.exists(compiled_file):
        shutil.move(compiled_file, os.path.join(OUTPUT_FOLDER, compiled_file))

    # Cleanup intermediate files
    os.remove(pyx_file)
    c_file = pyx_file.replace(".pyx", ".c")  # Get corresponding .c file
    if os.path.exists(c_file):
        os.remove(c_file)

# Remove temporary files
os.remove("temp_setup.py")
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("__pycache__", ignore_errors=True)

print(f"✅ All .py files have been compiled and saved in '{OUTPUT_FOLDER}'!")
