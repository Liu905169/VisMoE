model_name=VisLLM
train_epochs=50
learning_rate=1
# learning_rate=0.01
llama_layers=16

master_port=00097
num_process=1
batch_size=32
d_model=32
# d_ff=4
d_ff=128

comment='VisLLM-ETTh1-linear'
#  saves/ETTh1ls_12_h_s72_l12_clear \
python run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path dataset/ETT-small\
  --data_path ETTh1.csv \
  --adjdata dataset/data/sensor_graph/adj_mx.pkl \
  --model_id VisLLM_ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 332 \
  --label_len 96 \
  --pred_len 96 \
  --factor 3 \
  --enc_in 2 \
  --dec_in 2 \
  --c_out 2 \
  --des 'Exp' \
  --itr 1 \
  --d_model $d_model \
  --d_ff $d_ff \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment\
  --problems l_trend l_periods s_trend