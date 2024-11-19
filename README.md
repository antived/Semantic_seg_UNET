**WORK IN PROGRESS**

**PHASE 1** : Semantic segmentation of satellite data imagery to identlify complex and dense road networks from around the world.The identified roads are then used to construct a graph network comprising of nodes and roads.
**PHASE 2** A heuristic search algorithm like the A* algo or an alternate like Djikstra can then be used to find the optimal path between the nodes.

The current architecture being used for the means of semantic segmentation is a derived U-Net architecture which is currently using BCE loss but eventually will be using diceloss.

