import re
import base64
from cryptography.fernet import Fernet


class TreeNode:
    def __init__(self, label, error):
        self.label = label
        self.error = error
        self.left = None
        self.right = None
        self.condition = None

def ID3_classifier(samples, file_path = 'Inducer.dot', safe = False, key_path = None):
    def parse_dot(dot_string):
        # Extrai as definições de nós e relacionamentos
        node_defs = re.findall(r'node_(\d+) \[label="([^"]+)"\]', dot_string)
        edges = re.findall(r'node_(\d+)->node_(\d+) \[label="([^"]+)"\]', dot_string)
        
        nodes = {}
        
        # Extrai o label do nó e o erro de cada definiçãoe cria o TreeNode
        for node_def in node_defs:
            node_id, label_error = node_def
            split_label = label_error.split('\\n')
            
            if len(split_label) == 2:
                label, error = split_label
            else:
                label = split_label[0]
                error = None
            
            nodes[node_id] = TreeNode(label, error)

        # Extrai origem, destino, e condição para cada nó e sua ligação
        for edge in edges:
            src_id, tgt_id, label = edge
            if label.startswith('<='):
                nodes[src_id].left = nodes[tgt_id]
                nodes[src_id].condition = float(re.search(r'<= ([\d.]+)', label).group(1))
            else:
                nodes[src_id].right = nodes[tgt_id]

        return nodes['0']

    if safe: # quer dizer que deve aplicar a decodificação
        with open(key_path, 'rb') as key_file:
                key = key_file.read()
        cipher = Fernet(key)
        with open(file_path, 'rb') as file:
            encrypted_data = file.read()

        decrypted_data = cipher.decrypt(encrypted_data)
        # Abaixo se eu desejar escrever o arquivo decifrado
        #with open(output_file_name, 'wb') as file:
        #    file.write(decrypted_data)
        dot_content = decrypted_data.decode('utf-8')
    else:
        with open(file_path, 'r') as file:
                dot_content = file.read()

    #print(dot_content) - fins de debug
    root = parse_dot(dot_content)

    #print(root.label) - fins de debug
    #print(root.left.label) 
    #print(root.right.label) 

    def classificador(root, dicionario):
        #self.root = root # Recebe o nó raíz da árvore com toda ela construída
        #self.dicionario = dicionario # recebe dicionário onde chave é a frequência e valor é o psd
        # Agora o processamento para chegar na classificação

        pointer = root
        while(pointer.condition):
            if dicionario[pointer.label] <= pointer.condition:
                pointer = pointer.left
            else:
                pointer = pointer.right

        classif = pointer.label
        return classif

    classificacoes = []
    for ponto in samples:
        frequency_keys = [f"{i}Hz" for i in range(10, 2010, 10)]
        data_dictionary = {key: value for key, value in zip(frequency_keys, ponto)}
        classificacoes.append(classificador(root, data_dictionary))

    return classificacoes
