"""Validate the actual chapter 9 Notebook definitions, including PyG parity."""
import ast
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F


def definitions():
    path = Path(__file__).resolve().parents[1] / 'chap9图神经网络/图神经网络.ipynb'
    names = {'normalize_adj', 'MessagePassing', 'MeanConv', 'GCNLayer', 'GCN',
             'SAGELayer', 'GATLayer', 'GINLayer', 'GraphGIN', 'sample_neg_edges',
             'split_edges', 'decode', 'make_graphs', 'GINGraphClassifier', 'node_similarity'}
    ns = dict(torch=torch, nn=nn, F=F, np=np)
    for cell in json.loads(path.read_text(encoding='utf8'))['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns


def irregular_graph():
    # Irregular degrees expose bias placement; node 4 is isolated.
    a = torch.zeros(5, 5, dtype=torch.float64)
    for i, j in [(0, 1), (0, 2), (0, 3), (1, 2)]:
        a[i, j] = a[j, i] = 1
    return a


def test_source_target_direction_and_empty_max():
    ns = definitions()
    a = torch.zeros(3, 3, dtype=torch.float64)
    a[1, 0] = 1  # source 0 -> target 1
    x = torch.tensor([[-1e12], [3.], [7.]], dtype=torch.float64, requires_grad=True)
    for aggr in ['sum', 'mean', 'max']:
        actual = ns['MessagePassing'](aggr)(x, a)
        torch.testing.assert_close(actual, torch.tensor([[0.], [-1e12], [0.]], dtype=x.dtype))
        grad, = torch.autograd.grad(actual.sum(), x)
        torch.testing.assert_close(grad, torch.tensor([[1.], [0.], [0.]], dtype=x.dtype))


@pytest.mark.parametrize('kind', ['GCN', 'SAGE', 'GAT', 'GIN'])
def test_layers_match_pyg_forward_and_gradients(kind):
    pyg = pytest.importorskip('torch_geometric.nn')
    ns = definitions(); torch.manual_seed(12)
    a = irregular_graph()
    edge_index = a.nonzero().t().flip(0)  # dense row target -> PyG source first
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    xx = x.detach().clone().requires_grad_()
    layer = ns[kind+'Layer'](3, 4).double()
    if kind == 'GCN':
        native = pyg.GCNConv(3, 4).double()
        with torch.no_grad():
            native.lin.weight.copy_(layer.lin.weight); native.bias.copy_(layer.lin.bias)
        inp = ns['normalize_adj'](a)
        params = [(layer.lin.weight, native.lin.weight), (layer.lin.bias, native.bias)]
    elif kind == 'SAGE':
        native = pyg.SAGEConv(3, 4, normalize=False).double()
        with torch.no_grad():
            native.lin_r.weight.copy_(layer.lin.weight[:, :3])
            native.lin_l.weight.copy_(layer.lin.weight[:, 3:])
            native.lin_l.bias.copy_(layer.lin.bias)
        inp = a; params = [(layer.lin.bias, native.lin_l.bias)]
    elif kind == 'GAT':
        native = pyg.GATConv(3, 4, heads=1, concat=True, bias=False).double()
        with torch.no_grad():
            native.lin.weight.copy_(layer.W.weight)
            native.att_dst.copy_(layer.a[:4].view(1, 1, 4))
            native.att_src.copy_(layer.a[4:].view(1, 1, 4))
        inp = a; params = [(layer.W.weight, native.lin.weight)]
    else:
        native = pyg.GINConv(copy.deepcopy(layer.mlp), eps=0, train_eps=False).double()
        # GINConv initialization resets its MLP; align after construction.
        native.nn.load_state_dict(layer.mlp.state_dict())
        inp = a; params = list(zip(layer.mlp.parameters(), native.nn.parameters()))
    out = layer(x, inp); expected = native(xx, edge_index)
    torch.testing.assert_close(out, expected, atol=1e-11, rtol=1e-11)
    weight = torch.randn_like(out)
    (out*weight).sum().backward(); (expected*weight).sum().backward()
    torch.testing.assert_close(x.grad, xx.grad, atol=1e-11, rtol=1e-11)
    for p, q in params:
        torch.testing.assert_close(p.grad, q.grad, atol=1e-11, rtol=1e-11)
    if kind == 'SAGE':
        torch.testing.assert_close(layer.lin.weight.grad,
                                  torch.cat([native.lin_r.weight.grad, native.lin_l.weight.grad], dim=1),
                                  atol=1e-11, rtol=1e-11)
    if kind == 'GAT':
        torch.testing.assert_close(layer.a.grad,
                                  torch.cat([native.att_dst.grad.flatten(), native.att_src.grad.flatten()]),
                                  atol=1e-11, rtol=1e-11)


def test_node_equivariance_and_graph_invariance():
    ns = definitions(); torch.manual_seed(3)
    a = irregular_graph(); x = torch.randn(5, 3, dtype=a.dtype)
    order = torch.tensor([4, 2, 0, 3, 1])
    for name in ['GCNLayer', 'SAGELayer', 'GATLayer', 'GINLayer']:
        model = ns[name](3, 4).double()
        inp = ns['normalize_adj'](a) if name == 'GCNLayer' else a
        torch.testing.assert_close(model(x[order], inp[order][:, order]), model(x, inp)[order])
    graph = ns['GraphGIN'](3, 4).double().eval()
    torch.testing.assert_close(graph(x[order], a[order][:, order]), graph(x, a))


def test_link_splits_and_negative_sampling_do_not_leak_or_change_global_rng():
    ns = definitions()
    a = torch.zeros(10, 10)
    for i in range(10):
        a[i, (i+1)%10] = a[(i+1)%10, i] = 1
    np.random.seed(19); expected = np.random.rand(3)
    np.random.seed(19)
    train_a, train, val, test = ns['split_edges'](a)
    np.testing.assert_array_equal(np.random.rand(3), expected)
    sets = [set(map(tuple, edges.t().tolist())) for edges in [train, val, test]]
    assert len(set.union(*sets)) == 10 and sum(map(len, sets)) == 10
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    for edges in [val, test]:
        assert not train_a[edges[0], edges[1]].any() and not train_a[edges[1], edges[0]].any()
    gen = np.random.default_rng(1)
    held = ns['sample_neg_edges'](a, 4, rng=gen)
    blocked = a.bool().clone(); blocked[held[0], held[1]] = True
    sampled = ns['sample_neg_edges'](a, 20, exclude=blocked, rng=gen)
    assert (sampled[0] < sampled[1]).all() and not a[sampled[0], sampled[1]].any()
    assert not set(map(tuple, sampled.t().tolist())) & set(map(tuple, held.t().tolist()))
    with pytest.raises(ValueError): ns['sample_neg_edges'](a, 100)


def test_pyg_graph_batch_equals_individual_outputs_and_gradients():
    pyg = pytest.importorskip('torch_geometric.nn')
    data = pytest.importorskip('torch_geometric.data')
    ns = definitions(); ns.update(GINConv=pyg.GINConv, global_add_pool=pyg.global_add_pool)
    torch.manual_seed(8)
    graphs = []
    for n in [3, 5]:
        x = torch.randn(n, 3, dtype=torch.float64)
        edges = torch.tensor([[i for i in range(n-1)], [i+1 for i in range(n-1)]])
        graphs.append(data.Data(x=x, edge_index=torch.cat([edges, edges.flip(0)], 1)))
    model = ns['GINGraphClassifier'](3, 4, 2, dropout=0).double().eval()
    batch = data.Batch.from_data_list(graphs)
    out = model(batch.x, batch.edge_index, batch.batch)
    expected = torch.cat([model(g.x, g.edge_index, torch.zeros(g.num_nodes, dtype=torch.long)) for g in graphs])
    torch.testing.assert_close(out, expected, atol=1e-11, rtol=1e-11)
    p = tuple(model.parameters())
    actual_grads = torch.autograd.grad(out.square().sum(), p)
    expected_grads = torch.autograd.grad(expected.square().sum(), p)
    for a, b in zip(actual_grads, expected_grads):
        torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
