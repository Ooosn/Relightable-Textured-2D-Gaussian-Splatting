
echo "🟢 shell 脚本启动了"
which python

python render.py -m "C:\\Users\\namew\\Desktop\\beijing" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --load_num 400 \
                --write_images \
                --beijing_render \


: << option

计算fps 
--calculate_fps \

输出图片 
--write_images \

启动hgs
--use_hgs \

aaai渲染
--aaai_render \


option


