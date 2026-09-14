for i in `seq 0 10000 2120000`
do
    echo "running from $i to $((i + 10000))"
    python generate_facemaps.py -s $i -e $((i + 10000))
    echo "python generate_facemaps.py -s $i -e $((i + 10000)) done successfully"
done