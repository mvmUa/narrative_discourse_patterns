#!/bin/bash
# 
# The runcaevoraw command allows you to put a dir as an argument.
# When a file gave an error nothing was saved.
# The script allows to save discrete file executions.
# ./caevo_for_general.sh

#
# input is the file where the files to be processed are listed, with their absolute path
# 'DUC_Raw/duc2002-preproc/d095c/LA011889-0131'
# logFile is the file where "Success/error" for each file are stored
#
# sieve's are the execution logs that the program generates
#
# El archivo salida "1_The_Frog-King_,_or_Iron_Henry_.txt.info.xml" se guarda en el mismo lugar dd el de entrada
#

input="files_para_caevo"
logFile=log_big_tanda2_3.txt

while read file; do
	echo "Procesando $file" 
	nom_fake=`dirname $file | sed 's/\//_/g' `_`basename $file`
	( ./runcaevoraw.sh $file && echo "Exito procesando $file" >> $logFile      \
		&& mv sieve-output.xml sieves-ejecuciones/$nom_fake-sieve-output.xml ) || 	\
		  echo "Error procesando $file" >> $logFile 
 done < input


