# Implements a digraph
graph = {'a' : ['b', 'c'],
		 'b': ['c', 'd'],
		 'c': ['d'],
		 'd': ['c'],
		 'e': ['f'],
		 'f': ['c']
		}
def find_path(graph, start, end, path=[]):
	path = path + [start]
	if start == end:
		return path
	if start not in graph:
		return None
	for node in graph[start]:
		if node not in path:
			newpath = find_path(graph, node, end, path)
			if newpath: return newpath
	return None

path1 = find_path(graph, 'e', 'c')
print(path1)