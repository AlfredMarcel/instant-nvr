
# zjumocap上的实验脚本
# ================对应 bullet_show 文件夹内的参数==========================
# 使用例子   param1 为gpu编号  param2 为实验名

# sh scripts/just_try.sh "7," "baseline_1"

# 子弹时间demos

export GPUS=$1
en=$2

for name in 377
do
    python run.py --type bullet --cfg_file configs/bullet_show/inb_${name}.yaml exp_name ${en}_${name} gpus ${GPUS},
done

