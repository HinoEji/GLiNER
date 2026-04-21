python train.py --config train_config/phobert_large.yaml --path output/phobert_large
python eval.py --model "output/phobert_large" --batch_size 16 --log_dir "output/phobert_large/predict" --threshold 0.5
python eval.py --model "output/phobert_large" --batch_size 16 --log_dir "output/phobert_large/predict" --threshold 0.6
python eval.py --model "output/phobert_large" --batch_size 16 --log_dir "output/phobert_large/predict" --threshold 0.7
python eval.py --model "output/phobert_large" --batch_size 16 --log_dir "output/phobert_large/predict" --threshold 0.8
python eval.py --model "output/phobert_large" --batch_size 16 --log_dir "output/phobert_large/predict" --threshold 0.9