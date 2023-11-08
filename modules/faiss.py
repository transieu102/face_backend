import faiss
import numpy as np
import pickle
class Faiss:
    def __init__(self, path_to_index = None, vector_dim = 512) -> None:
        if not path_to_index:
            self.path_to_index = './database/data_auto.index'
            self.index = faiss.IndexFlatL2(vector_dim)
            self.map = []
            self.user = []
            self.vector_dim = vector_dim
        else:
            self.path_to_index = path_to_index
            self.index = faiss.read_index(path_to_index)
            with open(path_to_index.replace('.index','.pkl'), 'rb') as file:
                self.map = pickle.load(file, encoding='utf-8')
            with open(path_to_index.replace('.index','_user.pkl'), 'rb') as file:
                self.user = pickle.load(file, encoding='utf-8')
            self.vector_dim = vector_dim
    def add(self, features_vector, name):
        self.index.add(features_vector.detach().numpy())
        self.map.append(name)
        if name not in self.user:
            self.user.append(name)
    def search(self, features_vector, k):
        distances, indices = self.index.search(features_vector.detach().numpy(), k)
        if distances[0][0] > 0.6:
            return 'Not_Found' 
        result = {}
        for i in indices[0]:
            if self.map[i] not in result:
                result[self.map[i]] = 1
            else: 
                result[self.map[i]] += 1
        max_key = max(result, key=result.get)
        return max_key
    def save(self):
        faiss.write_index(self.index, self.path_to_index)
        with open(self.path_to_index.replace('.index','.pkl'), 'wb') as file:
            pickle.dump(self.map, file, protocol=pickle.HIGHEST_PROTOCOL)
        with open(self.path_to_index.replace('.index','_user.pkl'), 'wb') as file:
            pickle.dump(self.user, file, protocol=pickle.HIGHEST_PROTOCOL)