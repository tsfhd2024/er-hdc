python3 main.py \
    --x_train_filepath "../data/isolet/X_train.npy" \
    --y_train_filepath "../data/isolet/y_train.npy" \
    --x_test_filepath "../data/isolet/X_test.npy" \
    --y_test_filepath "../data/isolet/y_test.npy" \
    --dim 10000 \
    --epochs 200 \
    --lr 0.035 \
    --init_er 8 \
    --final_er 1 \
    --Thresh 10 \
    --encoding_system "kernel" \
    --nbr_cluster 1500 \
    --bootstrap 1.0 \
    --eps 1e-5 \
    --itr 2 \
    --weight_cluster 1.1

