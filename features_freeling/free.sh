#!/usr/bin/env bash

# To pass freeling to files in nested folders

# DIR : root directory
# OLD : name of the folder containing the texts
# NEW : name of the folder where the directory tree with the output will be reconstructed

# NEW=duc2002-preproc-free
#"""
DIR=headlines_for_freeling
NEW=headlines_from_freeling
OLD=headlines_from_freeling
j=0
function pasarFreeling() {

  for i in *
  do

    if test -f $i
    then
        horaIni=`date | cut -d ' ' -f4`

        echo "$horaIni $PWD\/$i $j - 894"

        start_time=$(date +%s)

        newpwd=`echo $PWD | sed "s/$OLD/$NEW/"`
        analyze -f en.cfg --outlv coref --ner --output conll < "$i"  > "$newpwd"/"$i".fre_out

        finish_time=$(date +%s)
        echo "$((finish_time - start_time))"

        #~ echo "$newName = $nameFile" >> $INDEX
        ((j++))

    elif test -d $i
    then
      newpwd=`echo $PWD | sed "s/$OLD/$NEW/"`

      if !( test -d $newpwd/$i)
        then
            mkdir $newpwd/$i
      fi

      cd $i
      pasarFreeling
      cd ..
    fi
  done
}


cd $DIR
pasarFreeling