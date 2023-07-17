### Scripts to run for exercises from C2 - c7 

#### C2
- Time measurement of code in C1 
- run 
    ```
    python lab2.py --question c2 --device cuda
    ```

#### C3
- I/O optimization starting from code in C2
- run 
    ```
    python lab2.py --question c3 
    ```

#### C4
- Profiling starting from code in C3
- run 
    ```
    python lab2.py --question c4 --num_workers 18 --device cuda  
    ```

#### C5
- Profiling starting from code in C3
- run 
    ```
    python lab2.py --question c5 --num_workers 18 --device cpu
    python lab2.py --question c5 --num_workers 18 --device cuda  
    ```
#### C6 
- Optimizers: SGD, SGD with Nesterov, Adagrad, Adadelta, and Adam.
- default is SGD, 5 epochs 
1. run SGD optimizer 
    ```
    python lab2.py --device cuda --question c6 --num_workers 18 --optimizer sgd 
    ```
2. run SGD with Nesterov
    ```
    python lab2.py --device cuda --question c6 --num_workers 18 --optimizer  nesterov
    ```
3. run adagrad
    ```
    python lab2.py --device cuda --question c6 --num_workers 18 --optimizer  adagrad
    ```
4. run adadelta
    ```
    python lab2.py --device cuda --question c6 --num_workers 18 --optimizer  adadelta
    ```
5. run adam
    ```
    python lab2.py --device cuda --question c6 --num_workers 18 --optimizer  adam
    ```

#### C7
- Experimenting without Batch Norm layer
- run following command to train the model 
    ```
    python lab2.py --question c7 --num_workers 18 --epochs 5 --device cuda 
    ```
- run following command to get the summary of the model
    ```
    python lab2.py --question c7 --model_summary True 
    ```