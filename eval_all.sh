#!/bin/bash

# Function to evaluate a model on a specific dataset
evaluate_model() {
    local model_name=$1
    local model_path=$2
    local config=$3
    local plane=$4
    local data_name=$5

    output_name="${model_name}_eval_${data_name}_${plane}"
    echo "Evaluating ${model_name} on ${data_name} data for ${plane}..."

    LD_PRELOAD=/lib64/libXrdPosixPreload.so:${LD_PRELOAD} python eval_merged.py --model ${model_path} --config ${config} --output ${output_name} --range 0 100 --gpu  --maskthreshold 0.5

}

# Define model paths
nomMC_U_model="/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/U_Plane/best_loss.pth"
nomMC_V_model="/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/nom/V_Plane/best_loss.pth"
randMC_U_model="/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/rand/U_Plane/best_loss.pth"
randMC_V_model="/scratch/7DayLifetime/abhat/wirecell/dnn_roi_icarus/training/mixed_samples/rand/V_Plane/best_loss.pth"


# Define config files
nomMC_U_config="config-nomMC_U.json"
nomMC_V_config="config-nomMC_V.json"
randMC_U_config="config-randMC_U.json"
randMC_V_config="config-randMC_V.json"


# Evaluation combinations

#  Evaluating nomMC models
evaluate_model "UNet_model_nomMC" ${nomMC_U_model} ${nomMC_U_config} "Plane0" "nomMC"
evaluate_model "UNet_model_nomMC" ${nomMC_V_model} ${nomMC_V_config} "Plane1" "nomMC"
evaluate_model "UNet_model_nomMC" ${nomMC_U_model} ${randMC_U_config} "Plane0" "randMC"
evaluate_model "UNet_model_nomMC" ${nomMC_V_model} ${randMC_V_config} "Plane1" "randMC"


# Evaluating randMC models
evaluate_model "UNet_model_randMC" ${randMC_U_model} ${nomMC_U_config} "Plane0" "nomMC"
evaluate_model "UNet_model_randMC" ${randMC_V_model} ${nomMC_V_config} "Plane1" "nomMC"
evaluate_model "UNet_model_randMC" ${randMC_U_model} ${randMC_U_config} "Plane0" "randMC"
evaluate_model "UNet_model_randMC" ${randMC_V_model} ${randMC_V_config} "Plane1" "randMC"




echo "All evaluations completed successfully."
