Launch examples:

- Sequentially:
python3 daura_driver.py -s md.dry.pdb -f md.reduced.xtc -n 10 -cutoff 0.14

- With PyCOMPSs:
runcompss --python_interpreter=python3 daura_driver.py -s md.dry.pdb -f md.reduced.xtc -n 10 -cutoff 0.14

- Help:
python3 daura_driver.py --help
