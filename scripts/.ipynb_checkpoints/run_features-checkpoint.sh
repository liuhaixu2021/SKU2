IFS=','

#for col in price comp_1_price comp_2_price comp_3_price comp_4_price
#do
#echo "Running with target: $col"
#python -u pre_same.py \
#    --data_path SKU_1.csv \
#    --target "$col"\
#    --pred_len 9
#done

for col in unitsordered sales cogs fba reffee adspend profit managed_fba_stock_level
do
echo "Running with target: $col"
python -u run_longExp.py \
    --model_id SKU_1 \
    --model LSTM_Linear \
    --data SKU_1 \
    --data_path SKU_1.csv \
    --features S \
    --target "$col"\
    --seq_len 90 \
    --label_len 30 \
    --pred_len 9
done

unset IFS

python csv_output.py