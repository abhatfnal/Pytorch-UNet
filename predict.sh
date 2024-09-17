
model=/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/bnb_cosmics/UNet/V_Plane/best_loss.pth

test=/exp/sbnd/data/users/abhat/wirecell_data/dnn_roi/ICARUS/81858535_45/tpc0_plane1_rec.h5


python predict.py -m ${model} --viz --no-save --no-crf --input ${test} --range 9 10 --mask-threshold 0.5