cd evrecs/LbL
 for f in $(ls -1 | sort -V); do python3 ~/Documents/tm/belle-ii/repo/LHE_NTuples/LHEConverter.py -i $f -o ~/Documents/tm/belle-ii/LbL/gen/${f:0:-4}.root; python3 ~/Documents/tm/belle-ii/repo/unpack.py ~/Documents/tm/belle-ii/LbL/gen/${f:0:-4}.root ~/Documents/tm/belle-ii/LbL/boosted/${f:0:-4}.root 1; python3 ~/Documents/tm/belle-ii/repo/smear.py ~/Documents/tm/belle-ii/LbL/boosted/${f:0:-4}.root ~/Documents/tm/belle-ii/LbL/smeared/${f:0:-4}.root; done
