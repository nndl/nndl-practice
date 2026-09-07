"""Execute the published Notebook definitions to protect masking and pair preprocessing."""
import ast
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


def notebook_definitions(part):
    path = Path(__file__).resolve().parents[1] / 'chap8注意力机制' / f'注意力机制-{part}.ipynb'
    namespace = dict(torch=torch, nn=nn, F=F, np=np, Dataset=Dataset, pad_sequence=pad_sequence)
    for cell in json.loads(path.read_text(encoding='utf8'))['cells']:
        if cell['cell_type'] != 'code':
            continue
        for node in ast.parse(''.join(cell['source'])).body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def test_sentence_pair_truncation_preserves_both_sides_and_separators():
    ns = notebook_definitions('下')
    vocab = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, 'a': 4, 'b': 5}
    ds = ns['LCQMCDataset']([('a'*100, 'bb', 1), ('', '', 0), ('a', 'b'*100, 0)], vocab, max_seq_len=8)
    for i in range(3):
        ids, segments, _ = ds[i]
        assert len(ids) <= 8 and ids[0] == 2 and ids[-1] == 3 and ids.count(3) == 2
        sep = ids.index(3)
        assert sep >= 2 and len(ids) - sep >= 3
        assert segments == [0]*(sep+1) + [1]*(len(ids)-sep-1)
    ids, segments, labels = ns['collate_fn']([ds[i] for i in range(3)])
    assert ids.shape == segments.shape == (3, 8) and labels.tolist() == [1, 0, 0]


def test_transformer_norms_independent_and_padding_invariant():
    ns = notebook_definitions('下')
    model = ns['Model_Transformer'](10, n_block=1, hidden_size=8, heads_num=2, intermediate_size=16,
                                     seq_len=16, hidden_dropout=0, attention_dropout=0).double().eval()
    ids = torch.tensor([[2, 4, 3, 5, 3, 0], [2, 4, 4, 3, 5, 3]])
    segments = torch.tensor([[0, 0, 0, 1, 1, 0], [0, 0, 0, 0, 1, 1]])
    a, b = model.layers[0].addnorm1, model.layers[0].addnorm2
    assert a.layer_norm.weight is not b.layer_norm.weight
    logits = model(ids, segments)
    torch.testing.assert_close(logits, model(F.pad(ids, (0, 3)), F.pad(segments, (0, 3))), atol=1e-11, rtol=1e-9)
    F.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert a.layer_norm.weight.grad is not None and b.layer_norm.weight.grad is not None


def test_book_attention_matches_native_projection_and_gradient():
    ns = notebook_definitions('上')
    torch.manual_seed(19)
    manual = ns['MultiHeadSelfAttention'](8, 2).double()
    native = nn.MultiheadAttention(8, 2, bias=False, batch_first=True).double()
    with torch.no_grad():
        native.in_proj_weight.copy_(torch.cat([manual.Q_proj.weight, manual.K_proj.weight, manual.V_proj.weight]))
        native.out_proj.weight.copy_(manual.out_proj.weight)
    x = torch.randn(2, 5, 8, dtype=torch.double, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    lens = torch.tensor([5, 2])
    a = manual(x, lens)
    b, _ = native(y, y, y, key_padding_mask=~ns['sequence_mask'](lens, 5, x.device))
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    probe = torch.randn_like(a)
    (a*probe).sum().backward(); (b*probe).sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-11, rtol=1e-9)
    torch.testing.assert_close(torch.cat([manual.Q_proj.weight.grad, manual.K_proj.weight.grad, manual.V_proj.weight.grad]), native.in_proj_weight.grad, atol=1e-11, rtol=1e-9)
    with pytest.raises(ValueError):
        manual(x, torch.tensor([5, 0]))


def test_rope_relative_positions_and_incremental_offsets():
    ns = notebook_definitions('下')
    freqs = ns['precompute_rope_freqs'](8, 30, dtype=torch.float64)
    rope = ns['apply_rope']
    q, k = [torch.randn(2, 4, 6, 8, dtype=torch.double) for _ in range(2)]
    a = rope(q, freqs, 7) @ rope(k, freqs, 9).transpose(-2, -1)
    b = rope(q, freqs, 0) @ rope(k, freqs, 2).transpose(-2, -1)
    torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-9)
    pieces = torch.cat([rope(q[:,:,:3], freqs, 7), rope(q[:,:,3:], freqs, 10)], dim=-2)
    torch.testing.assert_close(pieces, rope(q, freqs, 7))
    with pytest.raises(ValueError):
        rope(q, freqs, 27)


def test_reusable_full_mask_rows_zero_and_finite():
    from nndl.attention import scaled_dot_attention
    x = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[[True, False, False]], [[False, False, False]]])
    out, weights = scaled_dot_attention(x, x, x, mask)
    assert (out[1] == 0).all() and (weights[1] == 0).all()
    out.sum().backward()
    assert torch.isfinite(x.grad).all()
