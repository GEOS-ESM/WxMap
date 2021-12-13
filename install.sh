#!/bin/sh

install_path=`dirname $0`
INSTALL_PATH=`realpath $install_path`

cat modules | sed 's@$INSTALL_PATH@'$INSTALL_PATH@g > $INSTALL_PATH/utils/pyg_modules

exit 0
