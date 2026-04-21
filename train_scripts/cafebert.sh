python train.py --config train_config/cafebert_config.yaml --path output/cafebert
python eval.py --model "output/cafebert" --batch_size 16 --log_dir "output/cafebert/predict" --threshold 0.5
python eval.py --model "output/cafebert" --batch_size 16 --log_dir "output/cafebert/predict" --threshold 0.6
python eval.py --model "output/cafebert" --batch_size 16 --log_dir "output/cafebert/predict" --threshold 0.7
python eval.py --model "output/cafebert" --batch_size 16 --log_dir "output/cafebert/predict" --threshold 0.8
python eval.py --model "output/cafebert" --batch_size 16 --log_dir "output/cafebert/predict" --threshold 0.9