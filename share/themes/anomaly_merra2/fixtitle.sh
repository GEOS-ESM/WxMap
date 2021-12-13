#!/bin/sh

#export PATH=${PATH}:/discover/nobackup/projects/gmao/share/gmao_ops/Baselibs/v5.1.3_build1/x86_64-unknown-linux-gnu/ifort_18.0.3.222-mpt_2.17/Linux/bin

ODIR=/discover/nobackup/jardizzo/merra2_means

while read fname; do

  name=`basename $fname`
  oname=$ODIR/$name

  cp -p $fname $oname

  ncatted -a Title,global,m,c,'No Title' $oname

  echo $oname

done

exit 0
