script_abs=$(readlink -f "$0")
script_dir=$(dirname $script_abs)
cd ${script_dir}/
python setup.py build
cp build/lib*/*.so .

