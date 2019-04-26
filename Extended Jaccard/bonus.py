import networkx as nx


def jaccard_wt(graph, node):
  """
  The weighted jaccard score, defined above.
  Args:
    graph....a networkx graph
    node.....a node to score potential new edges for.
  Returns:
    A list of ((node, ni), score) tuples, representing the 
              score assigned to edge (node, ni)
              (note the edge order)
  """
  neighbors = set(graph.neighbors(node))  
  scores = []
  for n in graph.nodes():
  	if ((n not in neighbors) & (n != node)):
  		neighbors2 = set(graph.neighbors(n))  		
  		neu=0
  		deno1=0
  		deno2=0
  		for i in (neighbors & neighbors2):
  			neu+=(1/graph.degree(i))  		
  		for i in graph.neighbors(node):
  			deno1+=(graph.degree(i))  		
  		for i in graph.neighbors(n):
  			deno2+=(graph.degree(i))  				   		
  		scores.append(((node,n),neu/((1/deno1)+(1/deno2))))
  #print(sorted(scores, key=lambda x: (-x[1],x[0][1])))	
  return sorted(scores, key=lambda x: (-x[1],x[0][1]))

