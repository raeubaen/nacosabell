for i in $(seq 1001 5000); do
  cat LbL.dat | sed "s/#SEED/$RANDOM/g" > inputLbL.DAT
  echo $i
  timeout 90s ./superchic < inputLbL.DAT
  cp evrecs/evrectest.dat evrecs/LbL/evrec$i.dat
  source reco_1run_LbL.sh $i &
done
