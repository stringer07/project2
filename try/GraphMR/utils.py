import os
import time
import re
import ast
import torch
import pandas as pd

from graphviz import Digraph
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

MAX_TGT_LEN = 30

OPERATORS = {
    'Invert': '~',
    'Add': '+',
    'Sub': '-',
    'BitAnd': '&',
    'USub': '-',
    'BitOr': '|',
    'Mult': '*',
    'Pow': '**',
    'BitXor': '^',
    'Div': '/',
}

DEEPMIND_PATTERN_DICT= {
    'algebra__linear_1d': 
        r'Solve (.{1,}) = (.{1,}) for [a-z]',
    'arithmetic__add_or_sub':
        r'Work out (-?\d+\.?\d*) \+ (-?\d+\.?\d*)'
        + r'|Add together (-?\d+\.?\d*) and (-?\d+\.?\d*)'
        + r'|Put together (-?\d+\.?\d*) and (-?\d+\.?\d*)'
        + r'|Sum (-?\d+\.?\d*) and (-?\d+\.?\d*)'
        + r'|Total of (-?\d+\.?\d*) and (-?\d+\.?\d*)'
        + r'|Add (-?\d+\.?\d*) and (-?\d+\.?\d*)'
        + r'|What is (-?\d+\.?\d*) plus (-?\d+\.?\d*)'
        + r'|Calculate (-?\d+\.?\d*) \+ (-?\d+\.?\d*)'
        + r'|What is (-?\d+\.?\d*) \+ (-?\d+\.?\d*)'
        + r'|(-?\d+\.?\d*) - (-?\d+\.?\d*)'
        + r'|Work out (-?\d+\.?\d*) - (-?\d+\.?\d*)'
        + r'|What is (-?\d+\.?\d*) minus (-?\d+\.?\d*)'
        + r'|What is (-?\d+\.?\d*) take away (-?\d+\.?\d*)'
        + r'|Calculate (-?\d+\.?\d*) - (-?\d+\.?\d*)'
        + r'|What is (-?\d+\.?\d*) - (-?\d+\.?\d*)'
        + r'|(-?\d+\.?\d*) \+ (-?\d+\.?\d*)'
        + r'|(-?\d+\.?\d*)\+(-?\d+\.?\d*)',
    'arithmetic__add_sub_multiple':
        r'What is the value of (.{1,})'
        + r'|Evaluate (.{1,})'
        + r'|Calculate (.{1,})'
        + r'|What is (.{1,})'
        + r'|(.{1,})',
    'arithmetic__div':
        r'What is (.{1,}) divided by (.{1,})'
        + r'|Divide (.{1,}) by (.{1,})'
        + r'|Calculate (.{1,}) divided by (.{1,})'
        + r'|(.{1,}) divided by (.{1,})',
    'arithmetic__mixed':
        r'What is the value of (.{1,})'
        + r'|Evaluate (.{1,})'
        + r'|Calculate (.{1,})'
        + r'|What is (.{1,})'
        + r'|(.{1,})',
    'arithmetic__mul_div_multiple':
        r'What is the value of (.{1,})'
        + r'|Evaluate (.{1,})'
        + r'|Calculate (.{1,})'
        + r'|What is (.{1,})'
        + r'|(.{1,})',
    'arithmetic__mul':
        r'Calculate (.{1,})\*(.{1,})'
        + r'|Work out (.{1,}) \* (.{1,})'
        + r'|Multiply (.{1,}) and (.{1,})'
        + r'|Product of (.{1,}) and (.{1,})'
        + r'|What is the product of (.{1,}) and (.{1,})'
        + r'|What is (.{1,}) times (.{1,})'
        + r'|(.{1,}) times (.{1,})'
        + r'|(.{1,})\*(.{1,})'
        + r'|(.{1,}) \* (.{1,})',
    'polynomials__collect':
        r'Collect the terms in (.{1,})',
    'polynomials__expand':
        r'Expand (.{1,})',
    'polynomials__simplify_power':
        r'Simplify (.{1,}) assuming [a-zA-Z] is positive',
}

class ExprVisit(ast.NodeTransformer):
    '''
    Parsing an expression and generating the AST by the post-order traversal.
    For each expression, its variables must be a single character.
    '''
    def __init__(self, is_dag=True):
        self.is_dag = is_dag
        self.node_list = []
        self.edge_list = []
        self.subtree_memo = []

    def merge_node(self, node):
        '''Determine whether the current node can be merged'''
        same_node = None
        if self.subtree_memo is None:
            return same_node
        cur_node_type = 'binop' if hasattr(node, 'left') else 'unaryop'
        for p_node in self.subtree_memo:
            cur_p_node_type = 'binop' if hasattr(p_node, 'left') else 'unaryop'
            if cur_p_node_type != cur_node_type:
                continue
            elif cur_p_node_type == cur_node_type == 'binop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.left) == ast.dump(node.left) \
                    and ast.dump(p_node.right) == ast.dump(node.right):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)
            elif cur_p_node_type == cur_node_type == 'unaryop':
                if ast.dump(p_node.op) == ast.dump(node.op) \
                    and ast.dump(p_node.operand) == ast.dump(node.operand):
                    same_node = ast.dump(p_node.op) + str(p_node.col_offset)

        return same_node

    def visit_BinOp(self, node):
        '''
        Scaning binary operators, such as +, -, *, &, |, ^
        '''
        self.generic_visit(node)
        # node.col_offset is unique identifer
        node_str = ast.dump(node.op) + str(node.col_offset)
        if self.is_dag:
            same_node = self.merge_node(node)
            if same_node == None:
                self.subtree_memo.append(node)
                self.node_list.append(node_str)
            else:
                for idx in range(len(self.edge_list) - 1, -1, -1):
                    if node_str == self.edge_list[idx][0] or \
                        node_str == self.edge_list[idx][1]:
                        del self.edge_list[idx]
                node_str = same_node
        else:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_UnaryOp(self, node):
        '''
        Scaning unary operators, such as - and ~
        '''
        self.generic_visit(node)
        node_str = ast.dump(node.op) + str(node.col_offset)
        
        if self.is_dag:
            same_node = self.merge_node(node)
            if same_node == None:
                self.subtree_memo.append(node)
                self.node_list.append(node_str)
            else:
                for idx in range(len(self.edge_list) - 1, -1, -1):
                    if node_str == self.edge_list[idx][0] \
                        or node_str == self.edge_list[idx][1]:
                        del self.edge_list[idx]
                node_str = same_node
        else:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    def visit_Name(self, node):
        '''
        Scaning variables
        '''
        self.generic_visit(node)
        # node.col_offset will allocate a unique ID to each node
        node_str = node.id
        if not self.is_dag:
            node_str += '()' + str(node.col_offset)
        if node_str not in self.node_list:
            self.node_list.append(node_str)
        # If this node has parent, then create a edge between the node and its parent
        if hasattr(node.parent, 'op'):
            node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
            self.edge_list.append([node_str, node_parent_str])

        return node

    # Use visit_Constant when python version >= 3.8
    # def visit_Constant(self, node):
    def visit_Num(self, node):
        '''
        Scaning numbers
        '''
        self.generic_visit(node)
        node_str = str(node.n)
        if not self.is_dag:
            node_str += '()' + str(node.col_offset)
        if node_str not in self.node_list:
            self.node_list.append(node_str)
        node_parent_str = ast.dump(node.parent.op) + str(node.parent.col_offset)
        self.edge_list.append([node_str, node_parent_str])

        return node

    def get_result(self):
        return self.node_list, self.edge_list


def expr2graph(expr, is_dag=True):
    '''
    Convert a expression to a MSAT graph.

    Parameters:
        expr: A string-type expression.

    Return:
        node_list:  List for nodes in graph.
        edge_list:  List for edges in graph.
    '''
    ast_obj = ast.parse(expr)
    for node in ast.walk(ast_obj):
        for child in ast.iter_child_nodes(node):
            child.parent = node
    
    vistor = ExprVisit(is_dag=is_dag)
    vistor.visit(ast_obj)
    node_list, edge_list = vistor.get_result()

    return node_list, edge_list


def decompose(number, is_decimal):
    if len(number) == 1:
        return (number + '*0.1') if is_decimal else number
    expr, length = '', len(number)
    if not is_decimal:
        for i in range(length):
            if number[i] == '0': continue
            if i == length - 1: expr += '+' + number[i]
            elif i == length - 2: expr += ('' if length == 2 else '+') + number[i] + '*10'
            else: expr += ('+' if i else '') + number[i] + f'*10**{length-1-i}'
    else:
        for i in range(length):
            if number[i] == '0': continue
            elif i == 0: expr += number[i] + '*0.1'
            else: expr += ('+' if expr else '') + number[i] + f'*0.1**{i+1}'
    return expr

def repl(num):
    num = num.group(0)
    # num is integer 0 <= num <= 9
    if len(num) == 1: return num

    if '.' not in num:
        return  '(' + decompose(num, False) + ')'
    else:
        integer, decimal = num.split('.')
        if integer == '0':
            return '(' + decompose(decimal, True) + ')'
        else:
            return '(' + decompose(integer, False) + '+' + decompose(decimal, True) + ')'
def num_decompose(expr):
    return re.sub(r'\d+\.?\d*', repl, expr)


def expr_extract(dataset, qst):
    if dataset not in DEEPMIND_PATTERN_DICT:
        return qst
    search_obj = re.search(DEEPMIND_PATTERN_DICT[dataset], qst)

    if dataset == 'algebra__linear_1d':   
        expr = (search_obj.group(1) + '-(' + search_obj.group(2) + ')')
    elif dataset == 'arithmetic__add_or_sub':
        for i in range(1, len(search_obj.groups()) + 1, 2):
            if search_obj.group(i) != None:
                break
        op = '-' if i >= 23 else '+'
        expr = search_obj.group(i) + op + search_obj.group(i + 1)
    elif dataset == 'arithmetic__add_sub_multiple':
        for i in range(1, len(search_obj.groups()) + 1):
            if search_obj.group(i) != None:
                break
        expr = search_obj.group(i)
    elif dataset == 'arithmetic__div':
        for i in range(1, len(search_obj.groups()) + 1, 2):
            if search_obj.group(i) != None:
                break
        expr = search_obj.group(i) + '/' + search_obj.group(i + 1)
    elif dataset == 'arithmetic__mixed':
        for i in range(1, len(search_obj.groups()) + 1):
            if search_obj.group(i) != None:
                break
        expr = search_obj.group(i)
    elif dataset == 'arithmetic__mul_div_multiple':
        for i in range(1, len(search_obj.groups()) + 1):
            if search_obj.group(i) != None:
                break
        expr = search_obj.group(i)
    elif dataset == 'arithmetic__mul':
        for i in range(1, len(search_obj.groups()) + 1, 2):
            if search_obj.group(i) != None:
                break
        expr = search_obj.group(i) + '*' + search_obj.group(i + 1)
    elif dataset == 'polynomials__collect':
        expr = search_obj.group(1)
    elif dataset == 'polynomials__expand':
        expr = search_obj.group(1)
    elif dataset == 'polynomials__simplify_power':
        expr = search_obj.group(1)
    else:
        raise ValueError(f'{dataset} is not in dict.')

    expr = num_decompose(expr).replace(' ', '').replace('+-', '-').replace('--', '+')

    return expr


def dot_expr(nodes, edges):
    """
    Dot a graph G with graphviz.
    Parameters:
        nodes:  List, each element represents a node in G.
        edges:  List, each element is a two-tuples representing an directed edge in G.
    Returns:
        dot:    A Digraph object.
    """
    dot = Digraph()

    for node in nodes:
        label = node.split('()')[0]
        if label in OPERATORS:
            label = OPERATORS[label]
        dot.node(node, label)

    for edge in edges:
        dot.edge(edge[0], edge[1])

    dot.render(filename=str(time.time()), format='pdf')


class GraphExprDataset(InMemoryDataset):
    '''
    Base class of our dataset.
    '''
    def __init__(self, root, dataset, is_dag=True):
        self.is_dag = is_dag
        self.dataset = dataset
        self.qst_vocab = {}
        self.ans_vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2}
        if self.dataset != 'mba_simplify' and self.dataset != 'poly1': 
            self.qst_vocab['10'] = len(self.qst_vocab)
            self.qst_vocab['**'] = len(self.qst_vocab)
        if self.dataset == 'arithmetic__add_or_sub' or self.dataset == 'arithmetic__mul':
            self.qst_vocab['0.1'] = len(self.qst_vocab)
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return f'dataset/raw/{self.dataset}.csv'
    
    @property
    def processed_file_names(self):
        return f'{self.dataset}.pt'
    
    def download(self):
        pass

    def process(self):
        data_list = []

        df = pd.read_csv(self.raw_file_names, header=None, nrows=100000)
        
        max_tgt_len = float('-inf')
        
        # one sample per row
        for idx, row in tqdm(df.iterrows()):
            # row[0] is source expression, which will be transformed into a graph
            raw_qst, raw_ans = str(row[0]), str(row[1])
            expr = expr_extract(self.dataset, raw_qst)
            
            for c in expr:
                if c != '(' and c != ')' and c not in self.qst_vocab:
                    self.qst_vocab[c] = len(self.qst_vocab)
            for c in raw_ans:
                if c not in self.ans_vocab:
                    self.ans_vocab[c] = len(self.ans_vocab)

            x, edge_index = self._generate_graph(expr)
            # print(expr)

            # raw_ans is target, which will be represented as a one-hot vector
            # if len(raw_ans) <= MAX_TGT_LEN:
            y = [self.ans_vocab[c] for c in raw_ans]
            y.insert(0, self.ans_vocab['<sos>'])
            y.append(self.ans_vocab['<eos>'])

            max_tgt_len = max(max_tgt_len, len(y))

            y = torch.tensor(y, dtype=torch.long)

            # Fed the graph and the label into Data
            data = Data(x=x, edge_index=edge_index, y=y)
            # data = Data(x=x_s, edge_index=edge_index_s)
            data_list.append(data)

        # Pad the target
        for data in data_list:
            padding = torch.zeros(max_tgt_len - data.y.shape[0], dtype=torch.long)
            # padding = torch.tensor([self.ans_vocab['<pad>']] * (max_tgt_len - data.y.shape[0]), dtype=torch.long)
            data.y = torch.cat((data.y, padding), dim=0)
            padding = torch.zeros((data.x.shape[0], len(self.qst_vocab)-data.x.shape[1]), dtype=torch.float)
            data.x = torch.cat((data.x, padding), dim=1)

        self.max_tgt_len = max_tgt_len
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _generate_graph(self, expr):
        try:
            node_list, edge_list = expr2graph(expr, self.is_dag)
        except:
            raise ValueError(expr)

        node_feature = []

        for node in node_list:
            tag = node.split('()')[0]
            if tag in OPERATORS: tag = OPERATORS[tag]
            if tag == 'UAdd':
                exit(expr)

            feature = [0] * len(self.qst_vocab)
            feature[self.qst_vocab[tag]] = 1
            
            node_feature.append(feature)

        COO_edge_idx = [[], []]
        for edge in edge_list:
            s_node, e_node = node_list.index(edge[0]), node_list.index(edge[1])
            COO_edge_idx[0].append(s_node), COO_edge_idx[1].append(e_node)

        x = torch.tensor(node_feature, dtype=torch.float)
        edge_index = torch.tensor(COO_edge_idx, dtype=torch.long)

        return x, edge_index
