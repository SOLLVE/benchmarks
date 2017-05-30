make thyb INPUTTIME=1000 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb.log
make thyb INPUTTIME=1100 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb.log
make thyb INPUTTIME=1200 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb.log
make thyb INPUTTIME=1300 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb.log
make thyb INPUTTIME=1400 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb.log
make thyb INPUTTIME=1500 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb.log

make thyb INPUTTIME=1000 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb-dup.log
make thyb INPUTTIME=1100 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb-dup.log
make thyb INPUTTIME=1200 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb-dup.log
make thyb INPUTTIME=1300 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb-dup.log
make thyb INPUTTIME=1400 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb-dup.log
make thyb INPUTTIME=1500 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb-dup.log

make thyb INPUTTIME=1000 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb-hpf.log
make thyb INPUTTIME=1100 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb-hpf.log
make thyb INPUTTIME=1200 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb-hpf.log
make thyb INPUTTIME=1300 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb-hpf.log
make thyb INPUTTIME=1400 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb-hpf.log
make thyb INPUTTIME=1500 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb-hpf.log

make thyb INPUTTIME=1000 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb-pf2.log
make thyb INPUTTIME=1100 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb-pf2.log
make thyb INPUTTIME=1200 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb-pf2.log
make thyb INPUTTIME=1300 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb-pf2.log
make thyb INPUTTIME=1400 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb-pf2.log
make thyb INPUTTIME=1500 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb-pf2.log

make thyb INPUTTIME=1000 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb-dup-pf2.log
make thyb INPUTTIME=1100 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb-dup-pf2.log
make thyb INPUTTIME=1200 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb-dup-pf2.log
make thyb INPUTTIME=1300 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb-dup-pf2.log
make thyb INPUTTIME=1400 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb-dup-pf2.log
make thyb INPUTTIME=1500 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb-dup-pf2.log

make thyb INPUTTIME=1000 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1000-thyb-dup-hpf.log
make thyb INPUTTIME=1100 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1100-thyb-dup-hpf.log
make thyb INPUTTIME=1200 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1200-thyb-dup-hpf.log
make thyb INPUTTIME=1300 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1300-thyb-dup-hpf.log
make thyb INPUTTIME=1400 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1400-thyb-dup-hpf.log
make thyb INPUTTIME=1500 NVCCEXTRAFLAGS='-Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
./scripts/run_one.sh thyb &> log/cant-1500-thyb-dup-hpf.log

#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-100-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-200-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-300-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-400-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-500-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-600-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-700-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-800-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-900-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1000-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1100-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1200-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1300-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1400-um-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1500-um-pf2.log

#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-100-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-200-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-300-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-400-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-500-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-600-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-700-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-800-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-900-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1000-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1100-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1200-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1300-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1400-um-dup-pf2.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_DUPLICATE -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1500-um-dup-pf2.log

#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-100-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-200-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-300-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-400-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-500-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-600-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-700-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-800-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-900-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1000-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1100-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1200-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1300-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1400-um-dup.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_DUPLICATE' 2> /dev/null
#./scripts/run_one.sh um &> log/cant-1500-um-dup.log

#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1000-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1100-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1200-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1300-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1400-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_PREFETCH'
#source scripts/run_one.sh &> log/cant-1500-hyb-pf.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1000-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1000-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1000-hyb-dup.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1100-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1100-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1100-hyb-dup.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1200-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1200-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1200-hyb-dup.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1300-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1300-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1300-hyb-dup.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1400-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1400-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1400-hyb-dup.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1500-hyb-dup-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_PREFETCH -Xcompiler -DCUDA_UM_HOST_PREFETCH'
#source scripts/run_one.sh &> log/cant-1500-hyb-pf2.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500 -Xcompiler -DCUDA_UM_DUPLICATE'
#source scripts/run_one.sh &> log/cant-1500-hyb-dup.log

#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100'
#source scripts/run_one.sh &> log/cant-100-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200'
#source scripts/run_one.sh &> log/cant-200-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300'
#source scripts/run_one.sh &> log/cant-300-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400'
#source scripts/run_one.sh &> log/cant-400-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500'
#source scripts/run_one.sh &> log/cant-500-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600'
#source scripts/run_one.sh &> log/cant-600-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700'
#source scripts/run_one.sh &> log/cant-700-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800'
#source scripts/run_one.sh &> log/cant-800-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900'
#source scripts/run_one.sh &> log/cant-900-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000'
#source scripts/run_one.sh &> log/cant-1000-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100'
#source scripts/run_one.sh &> log/cant-1100-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200'
#source scripts/run_one.sh &> log/cant-1200-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300'
#source scripts/run_one.sh &> log/cant-1300-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400'
#source scripts/run_one.sh &> log/cant-1400-hyb.log
#make hyb NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500'
#source scripts/run_one.sh &> log/cant-1500-hyb.log

#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100'
#source scripts/run_one.sh &> log/cant-100-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200'
#source scripts/run_one.sh &> log/cant-200-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300'
#source scripts/run_one.sh &> log/cant-300-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400'
#source scripts/run_one.sh &> log/cant-400-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500'
#source scripts/run_one.sh &> log/cant-500-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600'
#source scripts/run_one.sh &> log/cant-600-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700'
#source scripts/run_one.sh &> log/cant-700-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800'
#source scripts/run_one.sh &> log/cant-800-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900'
#source scripts/run_one.sh &> log/cant-900-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000'
#source scripts/run_one.sh &> log/cant-1000-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100'
#source scripts/run_one.sh &> log/cant-1100-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200'
#source scripts/run_one.sh &> log/cant-1200-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300'
#source scripts/run_one.sh &> log/cant-1300-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400'
#source scripts/run_one.sh &> log/cant-1400-umpf.log
#make umpf NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500'
#source scripts/run_one.sh &> log/cant-1500-umpf.log

#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100'
#source scripts/run_one.sh &> log/cant-100-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200'
#source scripts/run_one.sh &> log/cant-200-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300'
#source scripts/run_one.sh &> log/cant-300-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400'
#source scripts/run_one.sh &> log/cant-400-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500'
#source scripts/run_one.sh &> log/cant-500-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600'
#source scripts/run_one.sh &> log/cant-600-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700'
#source scripts/run_one.sh &> log/cant-700-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800'
#source scripts/run_one.sh &> log/cant-800-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900'
#source scripts/run_one.sh &> log/cant-900-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000'
#source scripts/run_one.sh &> log/cant-1000-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100'
#source scripts/run_one.sh &> log/cant-1100-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200'
#source scripts/run_one.sh &> log/cant-1200-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300'
#source scripts/run_one.sh &> log/cant-1300-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400'
#source scripts/run_one.sh &> log/cant-1400-h.log
#make h NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500'
#source scripts/run_one.sh &> log/cant-1500-h.log

#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100'
#source scripts/run_one.sh &> log/cant-100-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200'
#source scripts/run_one.sh &> log/cant-200-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300'
#source scripts/run_one.sh &> log/cant-300-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400'
#source scripts/run_one.sh &> log/cant-400-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500'
#source scripts/run_one.sh &> log/cant-500-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600'
#source scripts/run_one.sh &> log/cant-600-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700'
#source scripts/run_one.sh &> log/cant-700-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800'
#source scripts/run_one.sh &> log/cant-800-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900'
#source scripts/run_one.sh &> log/cant-900-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000'
#source scripts/run_one.sh &> log/cant-1000-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100'
#source scripts/run_one.sh &> log/cant-1100-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200'
#source scripts/run_one.sh &> log/cant-1200-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300'
#source scripts/run_one.sh &> log/cant-1300-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400'
#source scripts/run_one.sh &> log/cant-1400-num.log
#make num NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500'
#source scripts/run_one.sh &> log/cant-1500-num.log

#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=100'
#source scripts/run_one.sh &> log/cant-100-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=200'
#source scripts/run_one.sh &> log/cant-200-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=300'
#source scripts/run_one.sh &> log/cant-300-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=400'
#source scripts/run_one.sh &> log/cant-400-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=500'
#source scripts/run_one.sh &> log/cant-500-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=600'
#source scripts/run_one.sh &> log/cant-600-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=700'
#source scripts/run_one.sh &> log/cant-700-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=800'
#source scripts/run_one.sh &> log/cant-800-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=900'
#source scripts/run_one.sh &> log/cant-900-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1000'
#source scripts/run_one.sh &> log/cant-1000-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1100'
#source scripts/run_one.sh &> log/cant-1100-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1200'
#source scripts/run_one.sh &> log/cant-1200-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1300'
#source scripts/run_one.sh &> log/cant-1300-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1400'
#source scripts/run_one.sh &> log/cant-1400-um.log
#make um NVCCEXTRAFLAGS='-Xcompiler -DINPUT_TIME=1500'
#source scripts/run_one.sh &> log/cant-1500-um.log
