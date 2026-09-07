"""Regression tests for the actual chapter 10 Notebook and shared model code."""
from pathlib import Path
import ast,json,sys,copy,math,string,io,contextlib
import torch
from torch import nn
from torch.nn import functional as F
from nndl.llm import NanoGPT,LoRALinear,sample_next,forward_cached,generate_cached,generate,apply_lora_to_model,CausalSelfAttention

def notebook_blocks():
    folder=Path(__file__).resolve().parents[1]/'chap10大语言模型与智能体'
    blocks={}
    for part in ['上','下']:
        for c in json.loads((folder/f'大语言模型与智能体-{part}.ipynb').read_text(encoding='utf8'))['cells']:
            idx=c.get('metadata',{}).get('book_block')
            if c['cell_type']=='code' and idx is not None:blocks[idx]=''.join(c['source'])
    return [blocks[i] for i in range(17)]

def definitions():
    ns=dict(torch=torch,nn=nn,F=F,math=math)
    for block in notebook_blocks():
        for n in ast.parse(block).body:
            if isinstance(n,(ast.ClassDef,ast.FunctionDef)):
                exec(compile(ast.Module(body=[n],type_ignores=[]),'notebook10','exec'),ns)
    return ns

def raises(fn):
    try:fn()
    except (ValueError,SyntaxError):return
    raise AssertionError('missing expected error')

def test_shared_model_definitions_match_notebook():
    source=Path(__file__).resolve().parents[1]/'nndl/llm.py'
    names={'CausalSelfAttention','FeedForward','Block','NanoGPT','LoRALinear','apply_lora_to_model','sample_next','generate'}
    def collect(blocks):
        return {n.name:ast.dump(n) for b in blocks for n in ast.parse(b).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names}
    assert collect(notebook_blocks())==collect([source.read_text(encoding='utf8')])

def test_causal_prefix_cache_and_greedy():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    m=NanoGPT(13,16,2,2,8,dropout=0).double().eval();x=torch.randint(13,(2,10));changed=x.clone();changed[:,6:]=torch.randint(13,(2,4))
    torch.testing.assert_close(m(x)[0][:,:6],m(changed)[0][:,:6],atol=1e-12,rtol=1e-12)
    whole=m(x)[0];parts=[];cache=None
    for left,right in [(0,3),(3,7),(7,10)]:
        out,cache=forward_cached(m,x[:,left:right],cache);parts.append(out)
    torch.testing.assert_close(torch.cat(parts,1),whole,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(generate(m,x[:,:3],6,temperature=0),generate_cached(m,x[:,:3],6))
    raises(lambda:generate_cached(m,x,8))

def test_attention_sdpa_forward_and_all_gradients():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    a=CausalSelfAttention(8,2,9).double();z=torch.randn(2,6,8,dtype=torch.float64,requires_grad=True)
    out=a(z);q,k,v=a.qkv(z).chunk(3,-1)
    q,k,v=[w.reshape(2,6,2,4).transpose(1,2) for w in (q,k,v)]
    expected=a.proj(F.scaled_dot_product_attention(q,k,v,is_causal=True).transpose(1,2).reshape(2,6,8))
    torch.testing.assert_close(out,expected,atol=1e-12,rtol=1e-12)
    params=(z,)+tuple(a.parameters());w=torch.randn_like(out)
    actual_grads=torch.autograd.grad((w*out).sum(),params,retain_graph=True)
    expected_grads=torch.autograd.grad((w*expected).sum(),params)
    for g,h in zip(actual_grads,expected_grads):torch.testing.assert_close(g,h,atol=1e-11,rtol=1e-11)

def test_sampling_exact_thresholds():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    prob=torch.tensor([[.5,.3,.2]],dtype=torch.float64).log().expand(100,3)
    assert (sample_next(prob,top_p=.5)==0).all()
    assert sample_next(torch.zeros(100,3),top_k=1).unique().numel()==1
    assert torch.equal(sample_next(prob,temperature=0),torch.zeros(100,1,dtype=torch.long))
    for kw in [{'temperature':-1},{'top_k':0},{'top_p':0},{'top_p':1.1}]:raises(lambda kw=kw:sample_next(prob,**kw))

def test_lora_dtype_gradients_and_merge():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    base=nn.Linear(5,3).double();original=copy.deepcopy(base.state_dict());adapter=LoRALinear(base,r=2)
    z=torch.randn(2,4,5,dtype=torch.float64);torch.testing.assert_close(adapter(z),base(z),atol=0,rtol=0)
    adapter(z).square().mean().backward();assert torch.count_nonzero(adapter.A.grad)==0 and adapter.B.grad.norm()>0
    assert base.weight.grad is None and base.bias.grad is None
    with torch.no_grad():adapter.B.add_(.03)
    merged=F.linear(z,base.weight+(adapter.B@adapter.A)*adapter.scaling,base.bias)
    torch.testing.assert_close(adapter(z),merged,atol=1e-12,rtol=1e-12)
    for k,v in base.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)

def test_sft_masking_and_dpo_gradients():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    chars=list(string.ascii_letters+' :\n');ns['stoi']={c:i for i,c in enumerate(chars)};ns['itos']=dict(enumerate(chars))
    model=NanoGPT(len(chars),32,1,2,8,dropout=0).double().eval();pairs=[('a:','ABC\n'),('word:','X\n')]
    xx,yy=ns['build_sft_batch'](pairs,max_len=32)
    assert (yy[0,:1]==-100).all() and (yy[1,:4]==-100).all()
    assert (yy!=-100).sum()==6
    batchloss=model(xx,yy)[1];weighted=0
    for pair in pairs:
        xp,yp=ns['build_sft_batch']([pair],max_len=32)
        weighted+=model(xp,yp)[1]*(yp!=-100).sum()
    torch.testing.assert_close(batchloss,weighted/6,atol=1e-12,rtol=1e-12)
    for pairs_bad in [[],[('','X')],[('a','')],[('a'*33,'B')]]:raises(lambda p=pairs_bad:ns['build_sft_batch'](p,max_len=32))
    
    ref=copy.deepcopy(model).requires_grad_(False).eval();loss=ns['dpo_loss'](model,ref,'a:','ABC\n','abc\n')
    torch.testing.assert_close(loss,torch.tensor(math.log(2),dtype=loss.dtype),atol=1e-12,rtol=1e-12)
    loss.backward();assert sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)>0
    assert all(p.grad is None for p in ref.parameters())
    xp,yp=ns['build_sft_batch']([('a:','ABC\n')]);lp=ns['logprob_of_sequence'](model,'a:','ABC\n')
    torch.testing.assert_close(lp,-model(xp,yp)[1]*4,atol=1e-12,rtol=1e-12)

def test_bounded_tools_and_grounded_retrieval():
    torch.manual_seed(0)
    ns=definitions();b=notebook_blocks()
    for block in b[15:17]:
        with contextlib.redirect_stdout(io.StringIO()):exec(block,ns)
    assert ns['calculator']('(15+27)*3')=='126'
    for expression in ['2**100','x+1','__import__("os")','1e999']:raises(lambda e=expression:ns['calculator'](e))
    assert '工具未能完成' in ns['react_loop']('计算 1 / 0')['answer']
    assert ns['react_loop']('x',model_fn=lambda q,h:'bad')['status']=='invalid'
    assert ns['react_loop']('x',max_steps=2,model_fn=lambda q,h:{'kind':'tool','name':'Search','input':'法国首都'})['status']=='limit'
    assert ns['rag']('GPU预约上限是多少？')['evidence'][0]['id']=='gpu'
    assert ns['rag']('明天天气如何？')['evidence']==[]
    raises(lambda:ns['retrieve']('x',k=0))
