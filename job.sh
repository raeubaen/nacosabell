#!/usr/bin/bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_105/x86_64-el9-gcc12-opt/setup.sh
cat /afs/cern.ch/user/r/rgargiul/belleii/LbL.dat | sed "s/#SEED/$RANDOM/g" > \
  /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/inputLbL.DAT
cd /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/
timeout 180s \
  ./superchic < /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/inputLbL.DAT
cp /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/evrecs/evrectest.dat \
   /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/evrecs/LbL/evrec$1.dat
source /afs/cern.ch/user/r/rgargiul/belleii/reco_LbL.sh $1
rm /afs/cern.ch/user/r/rgargiul/SuperChic/BUILD/bin/evrecs/LbL/evrec$1.dat
cd /afs/cern.ch/user/r/rgargiul/belleii
