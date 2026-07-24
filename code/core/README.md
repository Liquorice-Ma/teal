# HOW TO RUN THE CODE

### 1. Convert many trafic matrices files .npy into one .npz file to fit the format of this code
```bash
python3 main.py --scenario=prepare_synthetic_tm --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --scenario=prepare_synthetic_tm --traffic_type=synthetic --size_x=10 --size_y=10
python3 main.py --scenario=prepare_synthetic_tm --traffic_type=synthetic --size_x=15 --size_y=15
```

### 2. Using crop data to crop and create the topology
```bash
python3 main.py --scenario=crop_data --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --scenario=crop_data --traffic_type=synthetic --size_x=10 --size_y=10
python3 main.py --scenario=crop_data --traffic_type=synthetic --size_x=15 --size_y=15
```

### 3. Generate path list
```bash
python3 main.py --scenario=prepare_elephant_path_list --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --scenario=prepare_mice_path_list --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --scenario=prepare_mice_path_list_v2 --traffic_type=synthetic --size_x=5 --size_y=5
```

### 4. Run the experiment

##### a. Shortest path solver
```bash
python3 main.py --solver=sp --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --solver=se --traffic_type=synthetic --size_x=5 --size_y=5
python3 main.py --solver=ls --traffic_type=synthetic --size_x=5 --size_y=5
```

##### a.  S
python3 -W ignore main.py --solver=se

python3 -W ignore main.py --solver=spslp
python3 -W ignore main.py --solver=spselp

python3 -W ignore main.py --solver=spselp2

python3 -W ignore main.py --solver=spselpc
```
