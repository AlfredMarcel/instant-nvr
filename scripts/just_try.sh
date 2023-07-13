
# zjumocap上的实验脚本
# 使用例子   param1 为gpu编号  param2 为输出文件名 param3 为实验名

# sh scripts/just_try.sh "7," "baseline1.txt" "baseline_1"

export GPUS=$1

for name in 377 386 387 392 393 394
do
    python train_net.py --cfg_file configs/inb/inb_${name}.yaml exp_name $3_${name} gpus ${GPUS} silent True
    python run.py --type evaluate --cfg_file configs/inb/inb_${name}.yaml exp_name $3_${name} gpus ${GPUS} | grep -E 'mse:|psnr:|ssim:|lpips:'| awk -F ":" '{printf "%s,",$2}'  >> $2 
done

