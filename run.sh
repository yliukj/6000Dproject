device=0
data=HK/
batch=2
n_head=4
n_layers=1
d_model=32
d_rnn=32
d_inner=32
d_k=32
d_v=32
dropout=0.1
lr=1e-3
smooth=0.1
epoch=80

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$device python Main.py -data $data -batch $batch -n_head $n_head -n_layers $n_layers -d_model $d_model -d_rnn $d_rnn -d_inner $d_inner -d_k $d_k -d_v $d_v -dropout $dropout -lr $lr -smooth $smooth -epoch $epoch
