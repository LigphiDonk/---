import subprocess

# 定义三个命令
commands = [
    'conda activate ni && python experiments_server.py --model=simple-cnn --dataset=mnist --alg=fedavg --lr=0.1 --batch-size=64 --epochs=10 --n_parties=2 --mu=0.01 --rho=0.9 --comm_round=3 --partition=noniid-labeldir --beta=0.5 --device=cuda --noise=0 --sample=1 --init_seed=0',
    'conda activate ni && python experiments_client.py --comm_round=3 --client_id=0',
    'conda activate ni && python experiments_client.py --comm_round=3 --client_id=1'
]

# 启动三个终端并执行对应的命令
for command in commands:
    subprocess.Popen(command, shell=True)
