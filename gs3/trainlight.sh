date="250703_original"
subtask="NRHints"
model_root="E:/gsrelight-data"
model="Pixiu"



python train_light_direction.py -m $model_root/$subtask/$date/$model/ \
                --load_iteration -1 \
                --final_iteration 10000 \
                --skip_train \
                --skip_test \
                --use_nerual_phasefunc \
                --valid \
                --write_images \
                --view_num 1 \
                --load_num 1 \
                --test_iterations 2000 7000 10000 \
                --save_iterations 2000 7000 10000 \
                --checkpoint_iterations 20007000 10000 \
                --debug 