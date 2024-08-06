import numpy as np
import random
from os.path import exists
import matplotlib.pyplot as plt
import time
import networkx as nx
import os
game_name = "textmapworld_graphreasoning"

class SaveGraphInfo:

    def direction_list_maker(node, directions_list):
        from_node=[]
        to_node=[]
        opposite_direction_dict={'north':'south', 'south':'north', 'east':'west', 'west':'east'}
        for d in directions_list:
            if d[0]==node:
                from_node.append(d[1])
            elif d[2]==node:
                opposite_direction=opposite_direction_dict[d[1]]
                to_node.append(opposite_direction)

        combined=list(set(from_node) | set(to_node))
        return combined

    def get_directions(node, direction_list):
        node_directions = None  
        for i in direction_list:
            if i[0]==node:
                node_directions=i[1]
                break
        return node_directions

    def next_node_label(node, direction_list,nodes_list):
        dir2delta_inverse = {'north': np.array((0, 1)),
                    'south': np.array((0, -1)),
                    'east': np.array((1, 0)),
                    'west': np.array((-1, 0))}
        
        node_directions= SaveGraphInfo.get_directions(node, direction_list)
        next_nodes_list=[]
        for move in node_directions:
            next_node=tuple(np.array(node)+dir2delta_inverse[move])
            if next_node not in nodes_list:
                raise ValueError("The next chosen path is not possible")
            else:
                next_nodes_list.append((move,next_node))
        return next_nodes_list
    
    def get_node_directions(nodes_graph, paths):
        graph_directions = [(n, SaveGraphInfo.direction_list_maker(n, paths)) for n in nodes_graph]
        return graph_directions

    def get_moves_nodes_list(graph, graph_directions):
        moves_nodes_list = []
        for node in graph.nodes():
            node_dict = {
                "node": node,
                "node_moves": SaveGraphInfo.next_node_label(node, graph_directions, graph.nodes())
            }
            moves_nodes_list.append(node_dict)
        return moves_nodes_list

    


class GraphGenerator:
    
    def __init__(self, graph_type, n, m, n_rooms, cycle, ambiguity):
        self.n = n
        self.m = m
        self.n_rooms = n_rooms
        self.cycle = cycle
        self.map_array = np.zeros((n, m))
        self.graph_type = graph_type
        self.ambiguity = ambiguity

        if self.cycle!= "adding_cycle":
            self.G = nx.Graph()
            self.current_pos = np.random.randint(0, n), np.random.randint(0, m)
            self.map_array[self.current_pos] = 1
            self.G.add_node(self.current_pos)


    def generate_instance(self):

        dir2delta = {'north': np.array((0, 1)),
                     'south': np.array((0, -1)),
                     'east': np.array((1, 0)),
                     'west': np.array((-1, 0))}

        def find_cycle(source=None, orientation=None):

            if not self.G.is_directed() or orientation in (None, "original"):

                def tailhead(edge):
                    return edge[:2]

            elif orientation == "reverse":

                def tailhead(edge):
                    return edge[1], edge[0]

            elif orientation == "ignore":

                def tailhead(edge):
                    if edge[-1] == "reverse":
                        return edge[1], edge[0]
                    return edge[:2]

            explored = set()
            cycle = []
            final_node = None
            for start_node in self.G.nbunch_iter(source):
                if start_node in explored:
                    # No loop is possible.
                    continue

                edges = []
                # All nodes seen in this iteration of edge_dfs
                seen = {start_node}
                # Nodes in active path.
                active_nodes = {start_node}
                previous_head = None

                for edge in nx.edge_dfs(self.G, start_node, orientation):
                    # Determine if this edge is a continuation of the active path.
                    tail, head = tailhead(edge)
                    if head in explored:
                        # Then we've already explored it. No loop is possible.
                        continue
                    if previous_head is not None and tail != previous_head:
                        # This edge results from backtracking.
                        # Pop until we get a node whose head equals the current tail.
                        # So for example, we might have:
                        #  (0, 1), (1, 2), (2, 3), (1, 4)
                        # which must become:
                        #  (0, 1), (1, 4)
                        while True:
                            try:
                                popped_edge = edges.pop()
                            except IndexError:
                                edges = []
                                active_nodes = {tail}
                                break
                            else:
                                popped_head = tailhead(popped_edge)[1]
                                active_nodes.remove(popped_head)

                            if edges:
                                last_head = tailhead(edges[-1])[1]
                                if tail == last_head:
                                    break
                    edges.append(edge)

                    if head in active_nodes:
                        # We have a loop!
                        cycle.extend(edges)
                        final_node = head
                        break
                    else:
                        seen.add(head)
                        active_nodes.add(head)
                        previous_head = head

                if cycle:
                    break
                else:
                    explored.update(seen)

            else:
                assert len(cycle) == 0
                answer= "No cycle found"

            for i, edge in enumerate(cycle):
                tail, head = tailhead(edge)
                if tail == final_node:
                    break
            if len(cycle) != 0:
                answer=cycle[i:]
                
            return answer
        
        flag= False
        paths= []
        # check the cycle variable
        cycle_types=["cycle_true", "cycle_false", "random", "adding_cycle"]
        if self.cycle not in cycle_types:
            return "The cycle variable is not valid"
        
        while self.G.number_of_nodes() < self.n_rooms :
            # Prevent diagonal moves when cycle is set to "random"
            random_dir = np.random.choice(list(dir2delta.keys()))
            new_pos = tuple(np.array(self.current_pos) + dir2delta[random_dir])
            if min(new_pos) < 0 or new_pos[0] >= self.n or new_pos[1] >= self.m:
                # Illegal move
                continue

            if self.cycle != "random":
                # Initialize a copy of the graph to test whether it has a cycle
                copy_graph = self.G.copy()
                copy_graph.add_node(new_pos)
                copy_graph.add_edge(self.current_pos, new_pos)
                start_time = time.time()
                answer_cycle = find_cycle(copy_graph, orientation="ignore")
                # Check for cycle before adding an edge
                if self.cycle=="cycle_false" and answer_cycle != "No cycle found":
                    # Skip adding the edge that creates a cycle if Flag is True
                    if start_time > 8:
                        flag = True
                        break
                    continue
                if flag == True:
                    return "No graph generated"

            self.map_array[new_pos] = 1
            self.G.add_node(new_pos)
            self.G.add_edge(self.current_pos, new_pos)
            paths.append((self.current_pos, random_dir, new_pos))
            self.current_pos = new_pos

        # control time complexity
        if self.cycle=="cycle_true" and find_cycle(source=self.current_pos, orientation="ignore") == "No cycle found": 
            # this is the case graph does not contain but it should have one
            flag_2= False
            start_time = time.time()
            while start_time < 20:
                #print("start_time", start_time)
                for random_node in  list(self.G.nodes()):
                    for neighbor in self.G.neighbors(random_node):
                        if not self.G.has_edge(random_node, neighbor):
                            graph_copy = self.G.copy()
                            graph_copy.add_edge(neighbor, random_node)
                            check_cycle = find_cycle(graph_copy, orientation="ignore")
                            if check_cycle == "No cycle found":
                                flag_2 = False
                                continue
                            elif check_cycle != "No cycle found":
                                flag_2 = True
                                self.G.add_edge(neighbor, random_node)
                                break
            if flag_2 == False:
                return "No graph generated"
            
        elif self.cycle== "adding_cycle":
            # if we already have a cycle, we inform the user
            if find_cycle(source=self.current_pos, orientation="ignore") != "No cycle found":
                return "The graph already contains a cycle"
            else:
                # we add a cycle to the graph by adding an edge between two neibouring nodes
                for random_node in  list(self.G.nodes()):
                    for neighbor in self.G.neighbors(random_node):
                        if not self.G.has_edge(random_node, neighbor):
                            self.G.add_edge(neighbor, random_node)
                            break

        if len(list(self.G.nodes()))<self.n_rooms:
            return "No graph generated"
        if self.cycle=="cycle_false" and find_cycle(source=self.current_pos, orientation="ignore") != "No cycle found":
            return "No graph generated"
        
        graph_types=["named_graph", "unnamed_graph"]
        self.random_room = random.choice(list(self.G.nodes()))
        
        "--------------Graph type control--------------"
        if self.graph_type not in graph_types:
            return "The graph_type variable is not valid"
        
        if self.graph_type=="named_graph":

            def assign_types(ambiguity, graph):

                rooms = ["Bedroom", "Living room", "Kitchen", "Bathroom", "Dining room", "Study room", "Guest room", "Game room", "Home office", "Laundry room", "Pantry", "Attic", 
                                "Basement", "Garage", "Sunroom", "Mudroom", "Closet", "Library", "Foyer", "Nursery", "Home gym", "Media room", "Home theater", "Playroom", "Utility room", "Workshop", 
                                "Conservatory", "Craft room", "Music room", "Gallery", "Sauna", "Billiard room", "Bar", "Wine cellar", "Cellar", "Exercise room", "Studio", "Recreation room", "Solarium", "Balcony"]
                graph_unnamed_nodes = list(graph.nodes()) 
                total_nodes = len(graph.nodes())

                repetition_rooms, repetition_times = ambiguity[:2] if ambiguity!= None else (0, 0)
                if (repetition_rooms * repetition_times) > total_nodes:
                    return "The number of rooms is greater than the number of nodes in the graph"
                else:
                    # Generate a mapping from old node labels to room names for nodes specified by ambiguity
                    mapping = {}
                    if repetition_rooms > 0:
                        mapping = {}
                        repeated_nodes = random.sample(rooms, repetition_rooms)
                        for node in repeated_nodes:
                            rooms.remove(node)
                            for _ in range(repetition_times):
                                random_unnamed_node = random.choice(graph_unnamed_nodes)
                                mapping[random_unnamed_node] = node
                                graph_unnamed_nodes.remove(random_unnamed_node)

                    if len(graph_unnamed_nodes)>0:
                        # Randomly rename remaining nodes
                        random_rooms = random.sample(rooms, len(graph_unnamed_nodes))
                        for node in graph_unnamed_nodes:
                            mapping[node] = random.choice(random_rooms)
                            random_rooms.remove(mapping[node])

                    # Rename nodes using relabel_nodes function
                    if mapping:
                        if self.random_room in mapping:
                            if self.ambiguity != None:
                                self.random_room = f"{mapping[self.random_room]}_{self.random_room}"
                            else:
                                self.random_room = mapping[self.random_room]            
                    return mapping
            

        def save_graph_picture(graph):

            picture_number = random.randint(0, 10000)
            picture_name = "graph_" + str(picture_number) + ".png"
            # get the current working directory
            current_working_directory = str(os.getcwd().split("graph_generator.py")[0]) + "/games/"+ game_name+ "/resources/images/"
            file_exists = exists(current_working_directory + picture_name)
            
            if file_exists:
                picture_name = "graph_" + str(picture_number + 1) + ".png"

            if  self.graph_type=="unnamed_graph":
                nx.draw_networkx(graph, pos={n: n for n in graph.nodes()})
                # Display the graph
            elif self.graph_type=="named_graph":
                G_copy = self.G.copy()
                # Assign node labels using your function
                self.node_label_mapping = assign_types(self.ambiguity, G_copy)
                # Set node attributes
                nx.set_node_attributes(G_copy, self.node_label_mapping, 'label')
                # Get the 'label' attribute for each node
                labels = nx.get_node_attributes(G_copy, 'label')
                # Draw the graph with labels
                nx.draw_networkx(G_copy, pos={n: n for n in G_copy.nodes()}, labels=labels, with_labels=True)

            # Display the graph
            plt.savefig(current_working_directory + picture_name)
            plt.clf()

            return picture_name

        picture_name = save_graph_picture(self.G)
        graph_directions = SaveGraphInfo.get_node_directions(list(self.G.nodes()), paths)
        moves_nodes_list = SaveGraphInfo.get_moves_nodes_list(self.G, graph_directions)
        graph_dict={"Picture_Name":picture_name, "Graph_Type": self.graph_type, "Grid_Dimension": str(self.n), "Graph_Nodes":list(self.G.nodes()), "Graph_Edges": list(self.G.edges()), "N_edges": len(list(self.G.edges())) , "Initial_Position": self.random_room, "Directions": graph_directions, "Moves": moves_nodes_list ,"Cycle":self.cycle, 'Ambiguity': self.ambiguity}
        
        if self.graph_type=="named_graph":

            renamed_nodes=[]
            for node in list(self.G.nodes()):
                if self.ambiguity == None:
                    renamed_nodes.append(self.node_label_mapping[node])
                else:
                    renamed_nodes.append(f"{self.node_label_mapping[node]}_{node}")

            renamed_edges=[]
            for edge in list(self.G.edges()):
                if self.ambiguity == None:
                    renamed_edge=(self.node_label_mapping[edge[0]], self.node_label_mapping[edge[1]])
                else:
                    renamed_edge=(f"{self.node_label_mapping[edge[0]]}_{edge[0]}", f"{self.node_label_mapping[edge[1]]}_{edge[1]}")
                renamed_edges.append(renamed_edge)

            renamed_graph_directions= []
            for path in graph_directions:
                if self.ambiguity == None:
                    renamed_graph_directions.append((self.node_label_mapping[path[0]], path[1]))
                else:
                    renamed_graph_directions.append((f"{self.node_label_mapping[path[0]]}_{path[0]}", path[1]))


            renamed_moves_nodes_list=[]
            for move in moves_nodes_list:
                each_move = []
                if self.ambiguity == None:
                    renamed_node = self.node_label_mapping[move['node']]
                else:
                    renamed_node = f"{self.node_label_mapping[move['node']]}_{move['node']}"
                for movement in move['node_moves']:
                    if self.ambiguity == None:
                        change = self.node_label_mapping[movement[1]]
                    else:
                        change = f"{self.node_label_mapping[movement[1]]}_{movement[1]}"
                    each_move.append((movement[0], change))
                    renamed_moves_nodes_list.append({"node": renamed_node, "node_moves":each_move})
            
            graph_dict= {"Picture_Name":picture_name, "Graph_Type": self.graph_type, "Grid_Dimension": str(self.n), "Graph_Nodes":renamed_nodes, "Graph_Edges":renamed_edges , "N_edges": len(list(self.G.edges())) , "Initial_Position": self.random_room, "Directions": renamed_graph_directions , "Moves": renamed_moves_nodes_list ,"Cycle":self.cycle, 'Ambiguity': self.ambiguity, "Mapping": self.node_label_mapping}
        return  graph_dict
