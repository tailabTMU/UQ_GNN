import os

from datasets import MIMICDataset
import random
import pandas as pd
import argparse
import graphviz
from tqdm import tqdm


def main(args):
    cwd = os.getcwd()
    path = os.path
    pjoin = path.join

    dataset = MIMICDataset(root=pjoin(cwd, 'data', 'MIMICDataset'))

    raw_directory = dataset.raw_dir

    sample_path = pjoin(cwd, 'data', 'sample_mimic_graph')
    os.makedirs(sample_path, exist_ok=True)

    admissions_path = pjoin(raw_directory, 'admissions.csv')
    services_path = pjoin(raw_directory, 'services.csv')
    icu_stays_path = pjoin(raw_directory, 'icustays.csv')
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

    for idx in [10, 556]:
        patient_ids = admissions['subject_id'].unique().tolist()
        subject_id = patient_ids[idx]
        subject_admissions = admissions[admissions['subject_id'] == subject_id].sort_values(by=['in_date'])
        subject_services = services_df[services_df['subject_id'] == subject_id]
        subject_icu_stays = icu_stays[icu_stays['subject_id'] == subject_id].sort_values(by=['in_date'])

        subject_admission_service = pd.merge(subject_admissions, subject_services, on='hadm_id', how='left')

        first_graph = dataset[idx]

        edge_types = []
        for (_, edge_type, __) in first_graph.edge_types:
            if len(first_graph[edge_type].edge_index) > 0 and first_graph[edge_type].edge_index.size()[1] > 0:
                edge_types.append(edge_type)

        dot = graphviz.Digraph(comment=f'Dataset: {dataset}')
        node_colors = {
            'visit': '#fbbc04',
            'service': '#35a853',
        }
        for node_type in first_graph.node_types:
            with dot.subgraph() as subgraph:
                subgraph.attr(rank='same')
                for node in range(first_graph[node_type].num_nodes):
                    label = f"{node_type}_{node}"
                    subgraph.node_attr.update(style='filled', fontcolor='white',
                                              fillcolor=node_colors[node_type])
                    subgraph.node(f"{node_type}_{node}", label)

        edges = []
        for edge in range(first_graph['has_visit'].num_edges):
            first_node = int(first_graph['has_visit'].edge_index[0][edge])
            second_node = int(first_graph['has_visit'].edge_index[1][edge])
            edges.append(f'visit_{first_node},visit_{second_node},has_visit')
        for edge in range(first_graph['has'].num_edges):
            first_node = int(first_graph['has'].edge_index[0][edge])
            second_node = int(first_graph['has'].edge_index[1][edge])
            edges.append(f'visit_{first_node},service_{second_node},has')

        for edge_item in edges:
            first_node, second_node, edge_label = edge_item.split(',')
            dot.edge(str(first_node), str(second_node), str(edge_label))

        dot.render(
            pjoin(sample_path, f'patient_graph_{idx}'),
            # view=True,
            cleanup=True,
            format='pdf'
        )

        dot = graphviz.Digraph(comment=f'Dataset: {dataset}')
        node_colors = {
            'visit': '#808080',
            'service': '#C0C0C0',
        }
        for node_type in first_graph.node_types:
            with dot.subgraph() as subgraph:
                subgraph.attr(rank='same')
                for node in range(first_graph[node_type].num_nodes):
                    label = f"{node_type}_{node}"
                    subgraph.node_attr.update(style='filled', fontcolor='white',
                                              fillcolor=node_colors[node_type])
                    subgraph.node(f"{node_type}_{node}", label)

        edges = []
        for edge in range(first_graph['has_visit'].num_edges):
            first_node = int(first_graph['has_visit'].edge_index[0][edge])
            second_node = int(first_graph['has_visit'].edge_index[1][edge])
            edges.append(f'visit_{first_node},visit_{second_node},has_visit')
        for edge in range(first_graph['has'].num_edges):
            first_node = int(first_graph['has'].edge_index[0][edge])
            second_node = int(first_graph['has'].edge_index[1][edge])
            edges.append(f'visit_{first_node},service_{second_node},has')

        for edge_item in edges:
            first_node, second_node, edge_label = edge_item.split(',')
            dot.edge(str(first_node), str(second_node), str(edge_label))

        dot.render(
            pjoin(sample_path, f'patient_graph_{idx}-gray'),
            # view=True,
            cleanup=True,
            format='pdf'
        )

        subject_admission_service.to_csv(pjoin(sample_path, f'admission_service_{idx}.csv'), index=False)
        subject_icu_stays.to_csv(pjoin(sample_path, f'icu_stays_{idx}.csv'), index=False)

    print(f'Two Plots Stored in {sample_path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot MIMICDataset Graph')

    main(parser.parse_args())
