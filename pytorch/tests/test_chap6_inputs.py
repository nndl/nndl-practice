"""检查实际Notebook里的循环层、变长输入与共享适配器。"""
import ast
import json
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import Dataset
from nndl.rnn import NativeRecurrent
from nndl.text import tokenize, balanced_subset, build_vocab


def definitions(part):
    path=Path(__file__).resolve().parents[1]/f'chap6循环神经网络/循环神经网络-{part}.ipynb'
    ns=dict(torch=torch,nn=nn,Dataset=Dataset,tokenize=tokenize,
            pack_padded_sequence=pack_padded_sequence,pad_packed_sequence=pad_packed_sequence)
    for cell in json.loads(path.read_text(encoding='utf8'))['cells']:
        if cell['cell_type']=='code':
            nodes=[n for n in ast.parse(''.join(cell['source'])).body if isinstance(n,ast.ClassDef)]
            if nodes:exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns


@pytest.mark.parametrize('name,kind',[('SRN','rnn'),('LSTM','lstm')])
def test_actual_recurrence_forward_and_gradients(name,kind):
    torch.manual_seed(19);manual=definitions('上')[name](3,4).double();native=NativeRecurrent(manual,kind)
    x=torch.randn(2,5,3,dtype=torch.double,requires_grad=True);y=x.detach().clone().requires_grad_()
    a,b=manual(x),native(y);torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-10)
    probe=torch.randn_like(a);(a*probe).sum().backward();(b*probe).sum().backward()
    torch.testing.assert_close(x.grad,y.grad,atol=1e-12,rtol=1e-10)
    if kind=='rnn':pairs=[(manual.W.grad.T,native.cell.weight_ih_l0.grad),(manual.U.grad.T,native.cell.weight_hh_l0.grad),(manual.b.grad.flatten(),native.cell.bias_ih_l0.grad)]
    else:
        pairs=[]
        for prefix,ref in [('W_',native.cell.weight_ih_l0.grad),('U_',native.cell.weight_hh_l0.grad),('b_',native.cell.bias_ih_l0.grad)]:
            pairs.append((torch.cat([getattr(manual,prefix+g).grad.flatten() if prefix=='b_' else getattr(manual,prefix+g).grad.T for g in ('i','f','c','o')]),ref))
    for actual,expected in pairs:torch.testing.assert_close(actual,expected,atol=1e-12,rtol=1e-10)
    assert native.cell.bias_hh_l0.grad is None


def test_actual_bilstm_padding_and_empty_text():
    ns=definitions('下');torch.manual_seed(7);model=ns['Model_BiLSTM_FC'](8,4,5).double().eval()
    x=torch.tensor([[2,3,4,0],[5,6,0,0]]);lengths=torch.tensor([3,2])
    a=model(x,lengths)
    torch.testing.assert_close(a[:1],model(x[:1,:3],lengths[:1]),atol=1e-12,rtol=1e-10)
    torch.testing.assert_close(a,model(nn.functional.pad(x,(0,3)),lengths),atol=1e-12,rtol=1e-10)
    a.sum().backward();assert model.embedding_layer.weight.grad[0].abs().max()==0
    assert ns['IMDBDataset']([(' \n\t',1)],{'[PAD]':0,'[UNK]':1})[0][0]==[1]


def test_balanced_sampling_from_label_sorted_rows_and_train_only_vocab():
    rows=[(f'class{label} item{i}',label) for label in (0,1) for i in range(20)]
    picked=balanced_subset(rows,10,seed=3)
    assert picked==balanced_subset(rows,10,seed=3)
    assert len(picked)==len(set(picked))==10
    assert sum(label for _,label in picked)==5
    vocab=build_vocab([('training only training',0)],max_words=2)
    assert vocab=={'[PAD]':0,'[UNK]':1,'training':2,'only':3}
    assert vocab.get('validation',vocab['[UNK]'])==1
    with pytest.raises(ValueError):balanced_subset(rows,3)
