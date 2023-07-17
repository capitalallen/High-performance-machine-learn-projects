
## Instructions to run the python programs
### question 1
- run following command to start question 1 program 

    ```
    python q1_3.py
    ```
- Commands that I used to run the program with 1 GPUs
    ```
    sbatch run_q1_2.script
    ```
- different from commands used for question 2 
    ```
    #SBATCH --gres=gpu:rtx8000:1
    ```
### question 2
- run following command to start question 4 program 

    ```
    python q4.py
    ```
- 2 GPUs
    - change number of gpus to 2 in run_q1_2.script 
        ```
        #SBATCH --gres=gpu:rtx8000:2
        ```
    - Commands that I used to run the program with 2 GPUs 
        ```
        sbatch run_q1_2.script
        ```
- 4 GPUs
    - change number of gpus to 4 in run_q1_2.script 
        ```
        #SBATCH --gres=gpu:rtx8000:4
        ```
    - Commands that I used to run the program with 4 GPUs 
        ```
        sbatch run_q1_2.script
        ```

### question 4
- Accuracy when using large batch (8192)
- run following command to start question 4 program 

    ```
    python q4.py
    ```
- Commands that I used to run the program with 4 GPUs 
    ```
    sbatch run_q4.script
    ```
