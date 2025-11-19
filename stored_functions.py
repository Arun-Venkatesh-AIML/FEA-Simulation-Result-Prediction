import os
import re
import glob
import h5py
import numpy as np
import numpy as _np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.autograd as autograd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch.autograd as autograd

def parse_fem_file(fem_path):
    print(f"Parsing FEM file: {fem_path}")
    with open(fem_path, 'r') as file:
        lines = file.readlines()

    bulk_start = 0
    for i, line in enumerate(lines):
        if 'BEGIN BULK' in line.upper():
            bulk_start = i + 1
            break

    nodes = []
    elements = []
    materials = {}
    boundary_conditions = []
    loads = []

    for line in lines[bulk_start:]:
        line = line.strip()

        if line.startswith('GRID'):
            parts = line.split()
            try:
                node_id = int(parts[1])
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                nodes.append([node_id, x, y, z])
            except (ValueError, IndexError):
                continue

        elif line.startswith(('CHEXA','CPENTA')):
            parts = line.split()
            try:
                elem_id = int(parts[1])
                pid = int(parts[2])
                node_ids = [int(p) for p in parts[3:] if p.isdigit()]
                elements.append([elem_id, pid] + node_ids)
            except (ValueError, IndexError):
                continue

        elif line.startswith('MAT1'):
            line_fixed = re.sub(r'(?<=\d)(-\d)', r'E\1', line)
            parts = re.split(r'\s+', line_fixed.strip())
            try:
                mid = 1
                E_val = float(parts[1][1:])
                nu_val = float(parts[2])
                materials[mid] = {'E': E_val, 'nu': nu_val}
            except (ValueError, IndexError):
                continue

        elif line.upper().startswith('SPC'):
            parts = line.split()
            if len(parts) >= 4:
                try:
                    nid = int(parts[2])
                    comp_mask = parts[3].strip()
                    for ch in comp_mask:
                        dof = int(ch)
                        if dof <= 3:
                            boundary_conditions.append({'node': nid, 'dof': dof})
                except (ValueError, IndexError):
                    continue

        elif line.startswith('FORCE'):
            parts = line.split()
            try:
                nid = int(parts[2])
                fx = float(parts[4])
                fy = float(parts[5])
                fz = float(parts[6])
                loads.append({'node': nid, 'fx': fx, 'fy': fy, 'fz': fz})
            except (ValueError, IndexError):
                continue

    print(f"Parsed {len(nodes)} nodes")
    print(f"Parsed {len(elements)} 3D elements (CHEXA/CPENTA)")
    print(f"Parsed {len(materials)} materials")
    print(f"Parsed {len(boundary_conditions)} boundary conditions")
    print(f"Parsed {len(loads)} loads")

    node_df = pd.DataFrame(nodes, columns=['id','x','y','z'])
    return node_df, materials, boundary_conditions, loads

def load_simulation_output(pch_file_path):
    print(f"Loading displacement data from: {pch_file_path}")
    displacements = []
    capture = False

    with open(pch_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if '$DISPLACEMENTS' in line.upper() or 'DISPLACEMENT VECTOR' in line.upper():
                capture = True
                continue
            if capture and line.startswith('$'):
                continue
            if capture and line and not line.startswith('-'):
                parts = line.split()
                if len(parts) >= 5:
                    try:
                        nid = int(parts[0])
                        ux = float(parts[2]);  uy = float(parts[3]);  uz = float(parts[4])
                        displacements.append([nid, ux, uy, uz])
                    except ValueError:
                        continue
            if 'ENDDATA' in line.upper():
                break

    print(f"Extracted displacements for {len(displacements)} nodes.")
    df = pd.DataFrame(displacements, columns=['id', 'u', 'v', 'w'])
    return df

def export_results_to_h5(
    node_df,
    displacements_np,
    save_path='results/unseen_prediction.h5'
):
    n_nodes = len(node_df)
    assert displacements_np.shape == (n_nodes, 3), (
        f"Expected displacements_np shape ({n_nodes}, 3), got {displacements_np.shape}"
    )

    disp_dtype = np.dtype([
        ('ID',          'i8'),
        ('X',           'f8'),
        ('Y',           'f8'),
        ('Z',           'f8'),
        ('RX',          'f8'),
        ('RY',          'f8'),
        ('RZ',          'f8'),
        ('DOMAIN_ID',   'i8'),
    ])

    structured_data = np.zeros(n_nodes, dtype=disp_dtype)
    structured_data['ID'] = node_df['id'].values
    structured_data['X']  = displacements_np[:, 0]  # u_x
    structured_data['Y']  = displacements_np[:, 1]  # u_y
    structured_data['Z']  = displacements_np[:, 2]  # u_z
    structured_data['RX'] = 0.0
    structured_data['RY'] = 0.0
    structured_data['RZ'] = 0.0
    structured_data['DOMAIN_ID'] = 1

    domain_dtype = np.dtype([
        ('ID',          'i8'),
        ('SUBCASE',     'i8'),
        ('STEP',        'i8'),
        ('ANALYSIS',    'i8'),
        ('TIME_FREQ_EIGR','f8'),
        ('EIGI',        'f8'),
        ('MODE',        'i8'),
        ('DESIGN_CYCLE','i8'),
        ('RANDOM',      'i8'),
        ('SE',          'i8'),
        ('AFPM',        'i8'),
        ('TRMC',        'i8'),
        ('INSTANCE',    'i8'),
        ('MODULE',      'i8'),
    ])
    domain_data = np.array([
        (1, 1, 0, 10, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0, 0)
    ], dtype=domain_dtype)

    index_dtype = np.dtype([
        ('DOMAIN_ID','i8'),
        ('POSITION','i8'),
        ('LENGTH',  'i8'),
    ])
    index_data = np.array([
        (1, 0, n_nodes)
    ], dtype=index_dtype)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with h5py.File(save_path, 'w') as h5f:
        disp_grp = h5f.create_group("OPTISTRUCT/RESULT/NODAL")
        disp_grp.create_dataset(
            "DISPLACEMENT",
            data=structured_data,
            compression="gzip",
            compression_opts=9
        )

        res_grp = h5f["OPTISTRUCT/RESULT"]
        res_grp.create_dataset(
            "DOMAINS",
            data=domain_data,
            compression="gzip",
            compression_opts=9
        )

        index_grp = h5f.create_group("INDEX/OPTISTRUCT/RESULT/NODAL")
        index_grp.create_dataset(
            "DISPLACEMENT",
            data=index_data,
            compression="gzip",
            compression_opts=9
        )
    print(f"✅ HDF5 file written successfully: {save_path}")

# -----------------------------------------------------
# 1) PyTorch Dataset: NodesDataset (still builds raw_inputs/raw_targets)
# -----------------------------------------------------
class NodesDataset(Dataset):
    def __init__(self, fem_paths, pch_paths):
        self.raw_inputs = []
        self.raw_targets = []

        for fem_fp, pch_fp in zip(fem_paths, pch_paths):
            node_df, materials, boundary_conditions, loads = parse_fem_file(fem_fp)
            bc_nodes = set([bc['node'] for bc in boundary_conditions])

            x_coords = node_df['x'].values
            y_coords = node_df['y'].values
            z_coords = node_df['z'].values
            L = float(x_coords.max() - x_coords.min())
            B = float(y_coords.max() - y_coords.min())
            H = float(z_coords.max() - z_coords.min())

            assert len(materials) == 1
            mat = next(iter(materials.values()))
            E_val = float(mat['E'])
            nu_val = float(mat['nu'])

            assert len(loads) == 1
            load = loads[0]
            nid_f, fx, fy, fz = load['node'], load['fx'], load['fy'], load['fz']
            force_mag = float(np.sqrt(fx**2 + fy**2 + fz**2))

            load_coord = node_df[node_df['id'] == nid_f]
            assert len(load_coord) == 1
            xL, yL, zL = load_coord.iloc[0][['x', 'y', 'z']]

            disp_df = load_simulation_output(pch_fp)

            node_df_sorted = node_df.sort_values('id').reset_index(drop=True)
            disp_df_sorted = disp_df.sort_values('id').reset_index(drop=True)

            assert np.all(node_df_sorted['id'].values == disp_df_sorted['id'].values)

            for node_row, disp_row in zip(node_df_sorted.itertuples(index=False),
                                          disp_df_sorted.itertuples(index=False)):
                nid, x, y, z = node_row.id, node_row.x, node_row.y, node_row.z
                _, u, v, w = disp_row

                bc_flag = 1.0 if nid in bc_nodes else 0.0

                inp = [x, y, z, xL, yL, zL, force_mag, bc_flag, E_val, nu_val, L, B, H]
                tgt = [u, v, w]

                self.raw_inputs.append(inp)
                self.raw_targets.append(tgt)

            # Store metadata for physics loss
            self.node_ids = node_df_sorted['id'].tolist()
            self.force_dict = {nid_f: torch.tensor([fx, fy, fz], dtype=torch.float32)}
            self.bc_list = boundary_conditions

        _in = torch.tensor(self.raw_inputs, dtype=torch.float32)
        _out = torch.tensor(self.raw_targets, dtype=torch.float32)

        self.in_mean = _in.mean(dim=0)
        self.in_std = _in.std(dim=0)
        self.out_mean = _out.mean(dim=0)
        self.out_std = _out.std(dim=0)

        in_std_safe = torch.where(self.in_std == 0, torch.ones_like(self.in_std), self.in_std)
        out_std_safe = torch.where(self.out_std == 0, torch.ones_like(self.out_std), self.out_std)

        self.inputs = (_in - self.in_mean) / in_std_safe
        self.targets = (_out - self.out_mean) / out_std_safe

        print(f"✅ Dataset: {len(self.inputs)} samples built")
        print(f"   Inputs mean: {self.in_mean.tolist()}")
        print(f"   Inputs std:  {self.in_std.tolist()}")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ─────────────────────────────────────────────────────────────────────
# 2) Model Definition (input_dim=13)
# ─────────────────────────────────────────────────────────────────────
class ElasticNet(nn.Module):
    def __init__(self, in_dim=13, hidden=128, layers=8, out_dim=3):
        super().__init__()
        layers_list = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 1):
            layers_list += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers_list.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.net(x)