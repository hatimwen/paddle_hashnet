data_path='./datasets/COCO2014/'
batch_size=64

bit=16
python main_single_gpu.py \
--bit $bit \
--eval \
--ckp output/weights_$bit \
--data_path $data_path \
--batch_size $batch_size

bit=32
python main_single_gpu.py \
--bit $bit \
--eval \
--ckp output/weights_$bit \
--data_path $data_path \
--batch_size $batch_size

bit=48
python main_single_gpu.py \
--bit $bit \
--eval \
--ckp output/weights_$bit \
--data_path $data_path \
--batch_size $batch_size

bit=64
python main_single_gpu.py \
--bit $bit \
--eval \
--ckp output/weights_$bit \
--data_path $data_path \
--batch_size $batch_size
