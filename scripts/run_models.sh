IFS=','


for model in LSTM GRU RNN Informer Autoformer FEDformer LSTM_Linear DLinear NLinear PatchTST
do
echo "Running with target: $model"
python -u run_longExp.py \
    --model_id SKU_17_23 \
    --model "$model" \
    --data SKU_17_23 \
    --data_path SKU_17_23.csv \
    --features S \
    --target managed_fba_stock_level \
    --seq_len 24 \
    --label_len 24 \
    --pred_len 7
done

unset IFS

#python csv_output.py