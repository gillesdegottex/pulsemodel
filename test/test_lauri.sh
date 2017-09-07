VOICE="./slt_arctic_a0010"
MGCEP=mgcep # sptk program, give full path if not in $PATH
MGC2SP=mgc2sp

# f0 file is sptk style plain float binary
python ../analysis.py --inf0bin $VOICE".f0" --f0 $VOICE".f0" --spec $VOICE".spec" --nm $VOICE".bnm" --nm_nbbnds 25 $VOICE".wav"
$MGCEP -a 0.58 -m 39 -l 4096 -e 1.0E-08 -j 0 -f 0.0 -q 3 $VOICE".spec" > $VOICE".mgc"

# here any mgc can be used, e.g. from STRAIGHT or WORLD
$MGC2SP -a 0.58 -g 0.0 -m 39 -l 4096 -o 2 $VOICE".mgc" > $VOICE".synspec"
#python synthesis.py $VOICE".syn.wav" --f0file $VOICE".f0" --specfile $VOICE".synspec" --noisemaskfile $VOICE".nm" --noisemask_nbbnds 25 --fs 16000

python ../synthesis.py $VOICE".syn.wav" --f0file $VOICE".f0" --specfile $VOICE".synspec" --bndnmfile $VOICE".bnm" --fs 16000        
