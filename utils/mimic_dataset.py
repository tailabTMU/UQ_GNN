import torch
import torch.nn.functional as F
import os
import os.path as osp
import pandas as pd
import numpy as np
from torch_geometric.data import HeteroData
from dateutil import parser
import joblib
import torch.nn.functional as F
from tqdm import tqdm


def read_data(raw_directory, preprocessed_directory):
    all_preprocessed_files_exist = True
    if not osp.exists(osp.join(preprocessed_directory, f'adjacency_matrix_mimic.txt')):
        all_preprocessed_files_exist = False
    elif not osp.exists(osp.join(preprocessed_directory, f'nodes_mimic.txt')):
        all_preprocessed_files_exist = False
    elif not osp.exists(osp.join(preprocessed_directory, f'node_features_mimic.txt')):
        all_preprocessed_files_exist = False
    elif not osp.exists(osp.join(preprocessed_directory, f'graph_indicators_mimic.txt')):
        all_preprocessed_files_exist = False
    elif not osp.exists(osp.join(preprocessed_directory, f'graph_labels_mimic.txt')):
        all_preprocessed_files_exist = False

    if all_preprocessed_files_exist is False:
        create_files(raw_directory, preprocessed_directory)

    print("Loading Files")

    batch = read_file(osp.join(preprocessed_directory), '',
                      f'graph_indicators_mimic',
                      dtype=torch.long, to_tensor=True)
    node_features = read_file(osp.join(preprocessed_directory), '',
                              f'node_features_mimic',
                              dtype=torch.long)
    nodes = read_file(osp.join(preprocessed_directory), '', f'nodes_mimic')

    nodes = [[int(node[0]), node[1]] for node in nodes]
    node_types = {'service', 'visit'}
    print("Finding Service Nodes")
    service_node_indices = [index for index, node in enumerate(nodes) if node[1] == 'service']
    print("Finding Visit Nodes")
    visit_node_indices = [index for index, node in enumerate(nodes) if node[1] == 'visit']

    print("Service Features being Processed")
    # Service Features
    if (not osp.exists(
            osp.join(preprocessed_directory,
                     f'processed_services_mimic.pkl'))) or all_preprocessed_files_exist is False:
        service_features = np.array(
            [feature for index, feature in enumerate(node_features) if index in service_node_indices])
        service_features = cat([process_features(torch.tensor(service_features).to(torch.long).squeeze(), True), None])
        joblib.dump(service_features,
                    osp.join(preprocessed_directory,
                             f'processed_services_mimic.pkl'))
    else:
        print('Loading Saved File...')
        service_features = joblib.load(
            osp.join(preprocessed_directory, f'processed_services_mimic.pkl'))

    print("Visit Features being Processed")

    # Visit Features
    if (not osp.exists(
            osp.join(preprocessed_directory, f'processed_visits_mimic.pkl'))) or all_preprocessed_files_exist is False:
        visit_features = np.array(
            [feature for index, feature in enumerate(node_features) if index in visit_node_indices])
        visit_features_categorical_1 = process_features(torch.tensor(visit_features[:, 0]).to(torch.long).squeeze(),
                                                        True)
        visit_features_categorical_2 = process_features(torch.tensor(visit_features[:, 1]).to(torch.long).squeeze(),
                                                        True)
        visit_features = cat([
            visit_features_categorical_1,
            visit_features_categorical_2
        ])
        joblib.dump(visit_features,
                    osp.join(preprocessed_directory, f'processed_visits_mimic.pkl'))
    else:
        print('Loading Saved File...')
        visit_features = joblib.load(
            osp.join(preprocessed_directory, f'processed_visits_mimic.pkl'))

    y = read_file(osp.join(preprocessed_directory), '', f'graph_labels_mimic',
                  dtype=torch.long,
                  to_tensor=True)
    _, y = y.unique(sorted=True, return_inverse=True)

    edge_index = read_file(osp.join(preprocessed_directory), '',
                           f'adjacency_matrix_mimic',
                           dtype=torch.long, to_tensor=False)

    # Create HeteroData for each Patient
    print("Creating Dataset for Each Patient")
    dataset = []
    unique_graph_ids = torch.unique(batch).tolist()
    print(f'Total Graphs: {len(unique_graph_ids)}')
    processed_graphs = 0
    processed_visits = 0
    processed_services = 0
    for graph_id in tqdm(unique_graph_ids):
        processed_graphs += 1
        b = batch == graph_id
        indices = b.nonzero().squeeze().tolist()

        # Get Visit Nodes
        patient_visit_indices = [e for e in indices if e in visit_node_indices]
        # visit_feature_indices = [index for (index, e) in enumerate(patient_visit_indices) if e in visit_node_indices]
        # patient_visits = visit_features[visit_feature_indices]
        patient_visits = visit_features[processed_visits:processed_visits + len(patient_visit_indices)]
        processed_visits += len(patient_visit_indices)

        # Get Service Nodes
        patient_service_indices = [e for e in indices if e in service_node_indices]
        # service_feature_indices = [
        #     index for (index, e) in enumerate(patient_service_indices) if e in service_node_indices
        # ]
        # patient_services = service_features[service_feature_indices]
        patient_services = service_features[processed_services:processed_services + len(patient_service_indices)]
        processed_services += len(patient_service_indices)

        # Get Visit-Visit Edge Indices
        visit_visit_edge_index = [
            [patient_visit_indices.index(e[0]), patient_visit_indices.index(e[1])]
            for e in edge_index
            if (e[0] in patient_visit_indices and e[1] in patient_visit_indices)
        ]
        visit_visit_edge_index = process_edge_index(visit_visit_edge_index)
        # Get Visit-Service Edge Indices
        visit_service_edge_index = [
            [patient_visit_indices.index(e[0]), patient_service_indices.index(e[1])]
            for e in edge_index
            if (e[0] in patient_visit_indices and e[1] in patient_service_indices)
        ]
        visit_service_edge_index = process_edge_index(visit_service_edge_index)

        # Get Patient Graph Label
        patient_y = y[(graph_id - 1)]
        data = HeteroData()

        data['visit'].x = patient_visits
        data['service'].x = patient_services

        data['visit', 'has_visit', 'visit'].edge_index = visit_visit_edge_index
        data['visit', 'has_visit', 'visit'].edge_attr = np.array([])
        data['visit', 'has', 'service'].edge_index = visit_service_edge_index
        data['visit', 'has', 'service'].edge_attr = np.array([])

        data.y = patient_y

        dataset.append(data)

    return dataset


def create_files(raw_directory, preprocessed_directory):
    graph_indicator = []
    adjacency_matrix = []
    nodes = []
    graph_labels = []

    admissions_path = osp.join(raw_directory, 'admissions.csv')
    services_path = osp.join(raw_directory, 'services.csv')
    icu_stays_path = osp.join(raw_directory, 'icustays.csv')
    admissions_columns = ['subject_id', 'hadm_id', 'admittime', 'dischtime', 'admission_type', 'admission_location']
    services_columns = ['subject_id', 'hadm_id', 'curr_service']
    icu_stays_columns = ['subject_id', 'hadm_id', 'intime']

    admissions = pd.read_csv(admissions_path, usecols=admissions_columns)
    services_df = pd.read_csv(services_path, usecols=services_columns)
    icu_stays = pd.read_csv(icu_stays_path, usecols=icu_stays_columns)
    print('Find Missing Values...')
    nan_columns = admissions.columns[admissions.isna().any()].tolist()
    admissions.dropna(subset=nan_columns, inplace=True)
    nan_columns = services_df.columns[services_df.isna().any()].tolist()
    services_df.dropna(subset=nan_columns, inplace=True)
    nan_columns = icu_stays.columns[icu_stays.isna().any()].tolist()
    icu_stays.dropna(subset=nan_columns, inplace=True)

    no_service = 0
    indices = []
    print('Making sure all admissions have service info...')
    for idx in tqdm(range(len(admissions))):
        admission = admissions.iloc[idx]
        subject_id = admission['subject_id']
        hadm_id = admission['hadm_id']
        services = services_df[(services_df['subject_id'] == subject_id) & (services_df['hadm_id'] == hadm_id)]
        if len(services) == 0:
            no_service += 1
            indices.append(idx)
    print('Number of Removed Records (Admissions with no Service Info.):', no_service)
    admissions = admissions.drop(admissions.index[indices])

    admissions['in_date'] = pd.to_datetime(admissions['admittime']).dt.normalize()
    admissions['out_date'] = pd.to_datetime(admissions['dischtime']).dt.normalize()
    icu_stays['in_date'] = pd.to_datetime(icu_stays['intime']).dt.normalize()

    # Features for Each Node Type:
    #  1. VISIT: admission_type, admission_location
    #  2. SERVICE: curr_service

    # Turn words to number for categorical columns
    admissions_categorical_columns_num_classes = {}
    admissions_categorical_columns_min_class = {}
    for column in ['admission_type', 'admission_location']:
        unique_values = admissions[column].unique().tolist()
        admissions[column] = [unique_values.index(e) for e in admissions[column]]
        admissions_categorical_columns_num_classes[column] = len(unique_values)
        admissions_categorical_columns_min_class[column] = min(admissions[column])

    services_categorical_columns_num_classes = {}
    services_categorical_columns_min_class = {}
    for column in ['curr_service']:
        unique_values = services_df[column].unique().tolist()
        services_df[column] = [unique_values.index(e) for e in services_df[column]]
        services_categorical_columns_num_classes[column] = len(unique_values)
        services_categorical_columns_num_classes[column] = min(services_df[column])

    patient_ids = admissions['subject_id'].unique().tolist()
    graph_id_index = 1
    print('Creating Files for Each Patient...')
    for patient_id in tqdm(patient_ids):
        patient_data = admissions[admissions['subject_id'] == patient_id].sort_values(by=['in_date'])
        patient_services = services_df[services_df['subject_id'] == patient_id]
        patient_icu = icu_stays[icu_stays['subject_id'] == patient_id].sort_values(by=['in_date'])

        all_visit_in_dates = patient_data['in_date'].unique()
        # Keep the Last Visit for Label Only
        new_visit_in_dates = all_visit_in_dates
        last_visit_in_date = all_visit_in_dates[-1]
        if len(new_visit_in_dates) > 1:
            new_visit_in_dates = new_visit_in_dates[:-1]
            last_visit_in_date = new_visit_in_dates[-1]

        graph_label = 0  # No Admission
        if len(new_visit_in_dates) == 1 and len(patient_icu) == 0:
            graph_label = 0  # No Admission
        elif len(patient_icu) > 0:
            visit_data = patient_data[patient_data['in_date'] == last_visit_in_date].iloc[0]
            last_visit_out_date = visit_data['out_date']
            icu_date = patient_icu['in_date'].unique()[0]
            date1 = parser.parse(str(last_visit_out_date))
            date2 = parser.parse(str(icu_date))
            diff = date2 - date1
            if diff.days > 0 and diff.days <= 30:
                graph_label = 2  # ICU Admission
        else:
            visit_data = patient_data.iloc[len(new_visit_in_dates) - 1]
            last_visit_out_date = visit_data['out_date']
            date1 = parser.parse(str(last_visit_out_date))
            date2 = parser.parse(str(all_visit_in_dates[-1]))
            diff = date2 - date1
            if diff.days > 0 and diff.days <= 30:
                graph_label = 1  # Hospital Admission Visit


        graph_labels.append(int(graph_label))
        previous_visit = None
        services = {}
        for visit_index, visit_date in enumerate(new_visit_in_dates, start=1):
            visit_data = patient_data[patient_data['in_date'] == visit_date]
            visit_node_features = [
                visit_data.iloc[0]['admission_type'],
                visit_data.iloc[0]['admission_location']
            ]
            visit_node = Node(node_type='visit', node_features=visit_node_features)
            graph_indicator.append(graph_id_index)
            nodes.append(visit_node)
            if previous_visit is not None:
                adjacency_matrix.append([previous_visit.node_id, visit_node.node_id])

            services_data = patient_services[patient_services['hadm_id'] == visit_data.iloc[0]['hadm_id']]
            for _, row in services_data.iterrows():
                if f"service-{row['curr_service']}" in services:
                    service_node = nodes[services[f"service-{row['curr_service']}"]]
                else:
                    service_node_features = [
                        row['curr_service']
                    ]
                    service_node = Node(node_type='service', node_features=service_node_features)
                    graph_indicator.append(graph_id_index)
                    nodes.append(service_node)
                    services[f"service-{row['curr_service']}"] = len(nodes) - 1
                adjacency_matrix.append([visit_node.node_id, service_node.node_id])

            previous_visit = visit_node

        graph_id_index += 1

    if osp.exists(
            osp.join(preprocessed_directory, f'adjacency_matrix_mimic.txt')):
        os.remove(osp.join(preprocessed_directory, f'adjacency_matrix_mimic.txt'))
    with open(osp.join(preprocessed_directory, f'adjacency_matrix_mimic.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(x) for x in line]) for line in adjacency_matrix]))

    if osp.exists(osp.join(preprocessed_directory, f'nodes_mimic.txt')):
        os.remove(osp.join(preprocessed_directory, f'nodes_mimic.txt'))
    with open(osp.join(preprocessed_directory, f'nodes_mimic.txt'), 'w+') as f:
        f.write('\n'.join([','.join([str(node.node_id), str(node.type)]) for node in nodes]))

    if osp.exists(osp.join(preprocessed_directory, f'node_features_mimic.txt')):
        os.remove(osp.join(preprocessed_directory, f'node_features_mimic.txt'))
    with open(osp.join(preprocessed_directory, f'node_features_mimic.txt'),
              'w+') as f:
        f.write('\n'.join([','.join([str(feature) for feature in node.features]) for node in nodes]))

    if osp.exists(
            osp.join(preprocessed_directory, f'graph_indicators_mimic.txt')):
        os.remove(osp.join(preprocessed_directory, f'graph_indicators_mimic.txt'))
    with open(osp.join(preprocessed_directory, f'graph_indicators_mimic.txt'),
              'w+') as f:
        f.write('\n'.join([str(line) for line in graph_indicator]))

    if osp.exists(osp.join(preprocessed_directory, f'graph_labels_mimic.txt')):
        os.remove(osp.join(preprocessed_directory, f'graph_labels_mimic.txt'))
    with open(osp.join(preprocessed_directory, f'graph_labels_mimic.txt'),
              'w+') as f:
        f.write('\n'.join([str(line) for line in graph_labels]))


class Node:
    counter = 0

    def __init__(self, node_type: str, node_features=None):
        if node_features is None:
            node_features = []
        self.node_id = Node.counter
        self.type = node_type
        self.features = [int(x) for x in node_features]
        self._increment()

    @staticmethod
    def _encode_features(features, needs_to_be_encoded: bool):
        features = torch.tensor(features).to(torch.long).squeeze()
        if features.dim() == 1:
            features = features.unsqueeze(-1)
        features = features - features.min(dim=0)[0]
        features = features.unbind(dim=-1)
        if needs_to_be_encoded:
            features = [F.one_hot(e, num_classes=-1) for e in features]
        return torch.cat(features, dim=-1).to(torch.float)

    @staticmethod
    def _increment():
        Node.counter += 1

    def __str__(self):
        return f"Node {self.node_id}, a {self.type} node, with {len(self.features)} {'feature' if len(self.features) == 1 else 'features'}"


def process_edge_index(edge_index):
    starting_node = [edge[0] for edge in edge_index]
    ending_node = [edge[1] for edge in edge_index]
    edge_index = torch.tensor([starting_node,
                               ending_node], dtype=torch.long)
    # num_nodes = edge_index.max().item() + 1
    # edge_index, _ = remove_self_loops(edge_index)
    # edge_index, _ = coalesce(edge_index, None, num_nodes,
    #                          num_nodes)
    return edge_index


def process_features(features, is_categorical=False):
    if features.dim() == 1:
        features = features.unsqueeze(-1)
    if is_categorical:
        features = features - features.min(dim=0)[0]
        features = features.unbind(dim=-1)
        features = [F.one_hot(x, num_classes=-1) for x in features]
        features = torch.cat(features, dim=-1).to(torch.float)
    return features


def read_file(folder, prefix, name, to_number=None, dtype=None, to_tensor=False):
    path = osp.join(folder, '{}{}{}.txt'.format(prefix, '_' if prefix.strip() != '' else '', name))
    return read_txt_array(path, sep=',', dtype=dtype, to_tensor=to_tensor)


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1) if len(seq) > 0 else None


def read_txt_array(path, sep=None, start=0, end=None, dtype=None, to_tensor=False):
    to_number = None
    if dtype is not None and torch.is_floating_point(torch.empty(0, dtype=dtype)):
        to_number = float
    elif dtype is not None:
        to_number = int
    with open(path, 'r') as f:
        src = f.read().split('\n')

    if to_number is not None:
        src = [[to_number(x) for x in line.split(sep)[start:end]] for line in src]
    else:
        src = [[x for x in line.split(sep)[start:end]] for line in src]

    if to_tensor:
        src = torch.tensor(src).to(dtype).squeeze()
    return src
