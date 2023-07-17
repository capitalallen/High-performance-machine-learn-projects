# Lab 5

## Part A 

### Instruction to run Question 1
- I used "ptflops" python library to get total flops 
- call flop_count.py to get total flops with a GPU 
```
python flop_count.py
```
### Instruction to run Question 2

When using V100 GPU, Run 
```
python main_v100.py 
```
- It trains the model and stops when the testing accuracy hits 92%
- It saves model in a .pt file 

When using rtx2000 GPU, Run 
```
python main_rtx2000.py 
```
- It trains the model and stops when the testing accuracy hits 92%
- It saves model in a .pt file 

How I run the code to get the result 
```
chmod u+x ex_v100.sh 
chmod u+x ex_rxt2000.sh
chmod u+x run_q2_rxt2000.script
chmod u+x run_q2_v100.script
./ex_v100.sh
./ex_rxt2000.sh
```
- The shell script uses sbatch to run another script which trains and stores the model 

#### get predictions for collected images

- Since saved model is too large to upload directly, create a "models" folder in part1/ and download .pt files to that folder
    - https://drive.google.com/drive/folders/1IVUwjOd3PZtuadbVN8xMznNpqEs5qRxL?usp=sharing
- To get predictions, run 
```
python modeulUse.py
```

#### Measure the GPU utilization using nvidia-smi
- Run following commands 
```
chmod u+x watch_gpu.sh
./watch_gpu.sh
```
- store performance data to results_perf.csv
- runs two timeout command in the background 

### Instruction to run Question 3
1, C++ program
```
chmod u+x q3_1.sh
./q3_1.sh
```
- Go into q3_1.cpp to change M=1,5,10,50,100 for K=1,5,10,50,100
- call q3_1.sh after adjusting M value 
```
#define M 100 //1,5,10,50,100
#define N 1000000*M 
```

2, Use cudaMalloc()
```
chmod u+x q3_2.sh
./q3_2.sh
```
- adjust number of threads, number of blocks and K corresponding to each scenario in q3_2.cu 
```
#define BLOCKS 1 
#define THREADS 1
#define M 100 //1,5,10,50,100
#define N 1000000*M
```


3, use cudaMallocManaged()
```
chmod u+x q3_3.sh
./q3_3.sh
```
- adjust number of threads, number of blocks and K corresponding to each scenario in q3_3.cu
```
  int M = 100;
  int N = 1000000*M;
```
- adjust M to equal to 1, 5, 10, 50 or 100 


## Part B 
To get the result, run 
```
chmod u+x ex.sh 
./ex.sh
```