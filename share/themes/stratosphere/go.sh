#!/bin/sh

echo '<html>'
echo '<body bgcolor="#000000">'


echo '<ul>'

while read iname; do

  level=`echo $iname | cut -d'.' -f2`
  bname=`basename $iname`

  imtag='<img src="'$bname'" height="600" width="800">'
  echo '<li> <a href="https://portal.nccs.nasa.gov/datashare/gmao_ops/pub/fp/.internal/stratosphere/nps/'$iname'">'$imtag'</a></li>'

done

echo '</ul>'

echo '</body>'
echo '</html>'


exit 0
