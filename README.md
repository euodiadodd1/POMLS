# POMLS

The main body of code was based on the original source code for the paper available at: https://github.com/CharlieDinh/pFedMe

# Generating non-IID data
The baseline and non-IID data used in this experiment can be generated using the following commands. Each experiment must be run in turn to avoid files being overwritten. The data generated will be stored in pFedMe/data/train but must be moved to pFedMe/data/Mnist/. 

```
python data/Mnist/generate_baseline_20users.py
python data/Mnist/generate_label_skew_20users.py
python data/Mnist/generate_quantity_skew_20users.py
python data/Mnist/generate_concept_drift_20users.py
```
To modify the percentage used for concept drift, change this line in the file data/Mnist/generate_quantity_skew_20users.py
```
SHUFFLE_PERCENTAGE = 0.1
```
# Running experiments
Once data is generated and moved, the following command begins training.

```
python main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --personal_learning_rate 0.1 --beta 1 --lamda 10 --num_global_iters 800 --local_epochs 20 --algorithm pFedMe --numusers 5 --times 5
python main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --num_global_iters 800 --local_epochs 20 --algorithm FedAvg --numusers 5  --times 5
python main.py --dataset Mnist --model mclr --batch_size 20 --learning_rate 0.005 --beta 0.001  --num_global_iters 800 --local_epochs 20 --algorithm PerAvg --numusers 5  --times 5

```
# From the original documentation

All the train loss, testing accuracy, and training accuracy will be stored as h5py file in the folder "results". It is noted that we store the data for persionalized model and global of pFedMe in 2 separate files following format: DATASET_pFedMe_p_x_x_xu_xb_x_avg.h5 and DATASET_pFedMe_x_x_xu_xb_x_avg.h5 respectively (pFedMe for global model, pFedMe_p for personalized model of pFedMe, PerAvg_p is for personalized model of PerAvg).

To plot the results, change the variables in the main_plot.py file to the variables used in training and run main_plot.py

```

  numusers = 5
  num_glob_iters = 800
  dataset = "Mnist"
  local_ep = [20,20,20,20]
  lamda = [15,15,15,15]
  learning_rate = [0.005, 0.005, 0.005, 0.005]
  beta =  [1.0, 1.0, 0.001, 1.0]
  batch_size = [20,20,20,20]
  K = [5,5,5,5]
  personal_learning_rate = [0.1,0.1,0.1,0.1]
  algorithms = [ "pFedMe_p","pFedMe","PerAvg_p","FedAvg"]
  plot_summary_one_figure_mnist_Compare(num_users=numusers, loc_ep1=local_ep, Numb_Glob_Iters=num_glob_iters, lamb=lamda,
                             learning_rate=learning_rate, beta = beta, algorithms_list=algorithms, batch_size=batch_size, dataset=dataset, k = K, personal_learning_rate = personal_learning_rate)
 ``` 
 
