date="250314"
subtask="NRHints"
data_root="E:\\gs3_Dataset"
data_device="cpu"


python render.py -m "./output/$date/$subtask/Pikachu" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Cluttered" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Fish" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Cat" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Cat_on_Decor" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Cup-Fabric" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

python render.py -m "./output/$date/$subtask/Pixiu" \
                --load_iteration -1 \
                --skip_train \
                --skip_test \
                --valid \
                --use_nerual_phasefunc \
                --write_image

                