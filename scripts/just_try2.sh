
# zjumocap上的实验脚本
# ================对应 inb2 文件夹内的参数==========================
# 使用例子   param1 为gpu编号  param2 为实验名

# sh scripts/just_try.sh "7," "baseline_1"

# 脚本会保存实验指标到metrics文件夹

export GPUS=$1

en=$2
file="metrics/${en}.txt"

for name in 377
# 386 387 392 393 394
do
    python train_net.py --cfg_file configs/inb2/inb_${name}.yaml exp_name ${en}_${name} gpus ${GPUS} silent False
    python run.py --type evaluate --cfg_file configs/inb2/inb_${name}.yaml exp_name ${en}_${name} gpus ${GPUS} | grep -E 'mse:|psnr:|ssim:|lpips:'| awk -F ":" '{printf "%s,",$2}'  >> ${file} 
done

