
CUDA_VISIBLE_DEVICES='0,1' python3 transfer.py --model_type DA --dataset ABSA \
	--source=SQuAD --target=NaturalQuestions --test_batch_size=64 --n_labels=4 \
	--model_dir /home/chenxiang/bert-base-uncased/bert-base-uncased-pytorch_model.bin \
	--vocab_dir /home/chenxiang/bert-base-uncased/bert-base-uncased-vocab.txt \
	--cfg_dir /home/chenxiang/bert-base-uncased/bert-base-uncased-config.json