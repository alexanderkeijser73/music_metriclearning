import networkx as nx
import numpy as np

class SimilarityGraph():

    def __init__(self, df):
        self.df = df
        self.create_graph()

    def create_graph(self):
        self.graph = nx.DiGraph()
        self.id2path = {}
        print('starting preprocessing...')
        for _, row in self.df.iterrows():
            clip_ids, clip_votes, clip_paths = row[:3].values, row[3:6].values, row[6:].values

            # populate id2path dict
            for j, id in enumerate(clip_ids):
                if not id in self.id2path:
                    self.id2path[id] = clip_paths[j]

            idx = range(len(clip_votes))
            for i in idx:
                votes = clip_votes[i]
                if votes > 0:
                    odd_one_out_id = clip_ids[i]
                    other_idx = np.setdiff1d(idx, [i])
                    node1 = tuple(sorted(clip_ids[other_idx]))
                    node2 = tuple(sorted([clip_ids[other_idx][0], odd_one_out_id]))
                    node3 = tuple(sorted([clip_ids[other_idx][1], odd_one_out_id]))

                    # Find existing edge
                    ed = self.graph.get_edge_data(node1, node2)
                    if ed is not None:
                        edge_weight = ed['weight']
                        self.graph[node1][node2]['weight'] += votes
                    else:
                        self.graph.add_edge(node1, node2, weight=votes)

                    # Find existing edge
                    ed = self.graph.get_edge_data(node1, node3)
                    if ed is not None:
                        edge_weight = ed['weight']
                        self.graph[node1][node3]['weight'] += votes
                    else:
                        self.graph.add_edge(node1, node3, weight=votes)
        self.remove_inconsistencies()
        self.subgraphs = list(nx.weakly_connected_components(self.graph))
        max_size = max([len(sg) for sg in self.subgraphs])
        min_size = min([len(sg) for sg in self.subgraphs])
        referenced_clips = {clip_id for node in self.graph.nodes() for clip_id in node}
        print(f'graph consists of {len(self.subgraphs)} disjoint subgraphs containing between {min_size} and {max_size} vertices each')
        print(f'total graph contains {len(self.graph.edges)} edges/triplet constraints')
        print('number of referenced clips: ', len(referenced_clips))
        print('finished preprocessing.')


    def remove_inconsistencies(self):
        count = 0
        weight_points = 0
        for node in self.graph:
            to_remove = []
            for (u, v, d) in self.graph.edges(node, data=True):
                if self.graph.has_edge(v, u):
                    weight = d['weight']
                    weight_rev = self.graph.get_edge_data(v, u)['weight']

                    # If contradicting edges have equal votes, remove both
                    if weight == weight_rev:
                        to_remove.append((u, v))
                        to_remove.append((v, u))
                        count += 2
                        weight_points += weight * 2

                    elif weight > weight_rev:
                        to_remove.append((v, u))
                        self.graph[u][v]['weight'] = weight - weight_rev
                        count += 1
                        weight_points += 2 * weight_rev
                    elif weight < weight_rev:
                        to_remove.append((v, u))
                        self.graph[v][u]['weight'] = weight_rev - weight
                        count += 1
                        weight_points += 2 * weight
            self.graph.remove_edges_from(to_remove)

        print(f'removed {count} inconsistent edges.')
        print(f'removed {weight_points} weight points.')
        isolates = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(isolates)
        print(f'removed {len(isolates)} isolated nodes')

    @ staticmethod
    def triplet_from_edge(edge):
        all_ids = tuple({clip_id for node in edge for clip_id in node})
        odd_one_out_id = np.intersect1d(edge[0], edge[1])
        similar_pair = np.setdiff1d(all_ids, odd_one_out_id)
        triplet = np.concatenate((similar_pair, odd_one_out_id))
        return triplet


    # nodes_in_fold = lambda fold: [node for sg in fold for node in sg]
    # edges_in_fold = lambda graph, fold: [edges for node in nodes_in_fold(fold) for edges in self.graph.edges(node)]
    #
    # subgraphs = list(nx.weakly_connected_components(graph))
    # sg_sizes = set([len(sg) for sg in subgraphs])
    # referenced_clips = {clip_id for node in graph.nodes() for clip_id in node}
    # print(f'graph consists of {len(subgraphs)} disjoint subgraphs containing {sg_sizes} vertices each')
    # print(f'total graph contains {len(graph.edges)} edges/triplet constraints')
    # print('number of referenced clips: ', len(referenced_clips))
    # array = np.array(subgraphs)
    # np.random.shuffle(array)
    # splits = np.array_split(array, 10)
    # split_nodes = lambda split: [node for sg in split for node in sg]
    # split_edges = lambda split: [edges for node in split_nodes(split) for edges in graph.edges(node)]
    # constraints_per_split = [len(split_edges(split)) for split in splits]
    # print('average number of constraints per split: ', np.array(constraints_per_split).mean())