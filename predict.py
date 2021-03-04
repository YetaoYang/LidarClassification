import argparse
import json
import os
import sys
import time
import numpy as np
import open3d
import torch
from torch_geometric.nn import DataParallel, fps


ROOT_DIR = os.path.abspath(os.path.pardir)
sys.path.append(ROOT_DIR)
WORK_DIR = os.path.join(ROOT_DIR, 'data', 'Open3D')
sys.path.append(WORK_DIR)

from torch_geometric.utils import metric
from torch_geometric.data import Data, DataListLoader
from torch_geometric.datasets.Open3D import Open3DDataset
from cross_reference import Net

# Parser
parser = argparse.ArgumentParser()
parser.add_argument(
    "--num_samples",
    type=int,
    default=3000,
    help="# samples, each contains num_point points_centered",
)
parser.add_argument("--pt", default="", help="Checkpoint file")
parser.add_argument("--set", default="validation", help="train, validation, test")
flags = parser.parse_args()
hyper_params = json.loads(open(os.path.join(WORK_DIR, "vaihingen.json")).read())

# Create output dir
LOG_DIR = os.path.join(ROOT_DIR, hyper_params["logdir"])
output_dir = os.path.join(LOG_DIR, "result", "sparse")
os.makedirs(output_dir, exist_ok=True)

# Dataset
dataset = Open3DDataset(
    num_points_per_sample=hyper_params["num_point"],
    split=flags.set,
    box_size_x=hyper_params["box_size_x"],
    box_size_y=hyper_params["box_size_y"],
    use_color=hyper_params["use_color"],
    features=hyper_params["features"],
    path=WORK_DIR,
)

# Model
NUM_CLASSES = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(NUM_CLASSES,1)
model = DataParallel(model,device_ids=[0])
model.load_state_dict(torch.load(os.path.join(LOG_DIR, "best_model_epoch_185.pt"), map_location="cuda"))
model.to(device)
batch_size = 64
confusion_matrix = metric.ConfusionMatrix(NUM_CLASSES)

for semantic_file_data in dataset.list_file_data:
    print("Processing {}".format(semantic_file_data))

    # Predict for num_samples times
    points_collector = []
    pd_labels_collector = []
    # prob_collector = []

    # If flags.num_samples < batch_size, will predict one batch
    xs = []
    ys = []
    data_list = []
    ratio = 4 / dataset.num_points_per_sample
    points = torch.from_numpy(semantic_file_data.points).to(torch.float)
    idx = fps(points, ratio=ratio, random_start=False)
    centerpoints = points[idx]
    centerpoints = centerpoints.numpy()
    np.random.shuffle(centerpoints)
    num_batches = int(len(idx) / batch_size)

    for i in range(num_batches):
        sapmplepoints = centerpoints[i * batch_size: (i + 1) * batch_size]
        batch_data, batch_label, batch_raw = semantic_file_data.split_sample_batch(batch_size,
                                                                           dataset.num_points_per_sample,
                                                                           sapmplepoints, False, dataset.use_color,
                                                                           dataset.features)
        points_collector.extend(batch_raw)
        for j in range(batch_size):
            xs.append(torch.from_numpy(batch_data[j]).to(torch.float))
            ys.append(torch.from_numpy(batch_label[j]).to(torch.long))

    for (x, y) in zip(xs, ys):
        data = Data(pos=x[:, :3], x=x[:, 3:], y=y)
        data_list.append(data)
    data_loader = DataListLoader(data_list, batch_size)
    for data_list in data_loader:
        s = time.time()
        with torch.no_grad():
            out = model(data_list)
        pd_labels = out.max(dim=1)[1]
        print(
            "Batch size: {}, time: {}".format(batch_size, time.time() - s)
        )

        # Save to collector for file output
        pd_labels_collector.extend(pd_labels)
        # prob_collector.extend(np.array(out.cpu()))

        # Increment confusion matrix
        y = torch.cat([data.y for data in data_list]).to(out.device)
        for j in range(len(pd_labels)):
            confusion_matrix.increment(y[j].item(), pd_labels[j].item())

    # Save sparse point cloud and predicted labels
    file_prefix = os.path.basename(semantic_file_data.file_path_without_ext)
    sparse_points = np.array(points_collector).reshape((-1, 3))
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(sparse_points)
    pcd_path = os.path.join(output_dir, file_prefix + ".pcd")
    open3d.write_point_cloud(pcd_path, pcd)
    print("Exported sparse pcd to {}".format(pcd_path))

    sparse_labels = np.array(pd_labels_collector).astype(int).flatten()
    pd_labels_path = os.path.join(output_dir, file_prefix + ".labels")
    np.savetxt(pd_labels_path, sparse_labels, fmt="%d")
    print("Exported sparse labels to {}".format(pd_labels_path))

confusion_matrix.print_metrics()
