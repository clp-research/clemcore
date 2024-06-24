def load_nodes(nodes):
        """ transforms the nodes in the instance 
            from strings to tuples of ints"""
        loaded = []
        for node in nodes:
            without_brackets = node[1:-1]
            nums = without_brackets.split(',')
            tup = (int(nums[0].strip()), int(nums[1].strip()))
            loaded.append(tup)
        return loaded
    
def load_edges(edges):
    """ transforms the edges in the instance 
        from strings to tuples of tuples of ints"""
    loaded = []
    for edge in edges:
        edge = edge.replace('(', '')
        edge = edge.replace(')', '')
        nums = edge.split(',')
        tup1 = (int(nums[0].strip()), int(nums[1].strip()))
        tup2 = (int(nums[2].strip()), int(nums[3].strip()))
        loaded.append((tup1, tup2))
    return loaded

def load_imgs(imgs):
    """ changes the keys for images from strings to 
        tuples of ints """
    loaded = {}
    for key, value in imgs.items():
        key_tup = load_nodes([key])[0]
        loaded[key_tup] = value
    return loaded

def load_cats(cats):
    loaded = {}
    for key, value in cats.items():
        key_tup = load_nodes([key])[0]
        loaded[key_tup] = value
    return loaded

def load_start(start):
    """ changes the starting node from string to a
        tuple of ints """
    tup = load_nodes([start])[0]
    return tup
        
def load_instance(instance):
    """The instance has been serialized using string for all the 
    tuples. This function reverts this process by transforming the 
    strings used to represent the graph as tuples of ints, so they
    can be worked with. 

    Args:
        instance (dict): the current instance
    """
    loaded_nodes = load_nodes(instance['nodes'])
    loaded_edges = load_edges(instance['edges'])
    loaded_imgs = load_imgs(instance['imgs'])
    loaded_cats = load_cats(instance['cats'])
    loaded_start = load_start(instance['start'])
    loaded_target = load_start(instance['target'])
    
    return {
        'nodes': loaded_nodes,
        'edges': loaded_edges,
        'imgs': loaded_imgs,
        'start': loaded_start,
        'cats': loaded_cats,
        'target': loaded_target
    }
    
def edge_to_delta(edge):
        dx = edge[1][0] - edge[0][0]
        dy = edge[1][1] - edge[0][1]
        return (dx, dy)