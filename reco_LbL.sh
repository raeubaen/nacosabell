python3 /afs/cern.ch/user/r/rgargiul/belleii/LHE_NTuples/LHEConverter.py \
  -i /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/evrecs/LbL/evrec$1.dat \
  -o /eos/user/r/rgargiul/LbL_belle2/gen/evrec$1.root

python3 /afs/cern.ch/user/r/rgargiul/belleii/unpack.py \
    /eos/user/r/rgargiul/LbL_belle2/gen/evrec$1.root \
    /eos/user/r/rgargiul/LbL_belle2/boosted/evrec$1.root 1

python3 /afs/cern.ch/user/r/rgargiul/belleii/smear.py \
    /eos/user/r/rgargiul/LbL_belle2/boosted/evrec$1.root \
    /eos/user/r/rgargiul/LbL_belle2/smeared/evrec$1.root
