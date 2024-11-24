# 👋 HiMoE: Heterogeneity-Informed Mixture of Experts for Fair Spatial-Temporal Forecasting

python main.py --conf .\conf\himoe.json --dataset PEMS04 --data_dir .\data\PEMS04\ --gpuid 0 --mode test_only --best_model_path .\exp\pems04-best.pkl

python main.py --conf .\conf\himoe.json --dataset KNOWAIR --data_dir .\data\KNOWAIR\ --gpuid 0 --mode test_only --best_model_path .\exp\knowair-best.pkl