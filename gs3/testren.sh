date="250703_original"
subtask="NRHints"
model_root="E:/gsrelight-data"
model="Pixiu"

python render.py -m $model_root/$subtask/$date/$model/ \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --use_nerual_phasefunc \
                --valid \
                --write_images \
                --offset 0.1 \
                --load_num 400 \
                --force_save \
                --shadowmap_render
                        
: <<EOF

--gamma

                --calculate_fps \
--synthesize_video  


                
EOF





