echo "Running test.sh"

echo "1 sphere:"
./program ray-1.txt

wait

echo "150 spheres:"
./program ray-150.txt

wait

echo "300 spheres:"
./program ray-300.txt

wait

echo "625 spheres:"
./program ray-625.txt

wait

echo "1250 spheres:"
./program ray-1250.txt

wait

echo "2500 spheres:"
./program ray-2500.txt

wait

echo "5000 spheres:"
./program ray-5000.txt

wait

echo "10000 spheres:"
./program ray-many.txt