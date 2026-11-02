// Run by tests/test_frontend.py: executes web/index.html's real script against a stub DOM.
const fs=require('fs');const jobs=JSON.parse(fs.readFileSync(process.env.JOBS_JSON));
const els={};const mk=id=>els[id]||(els[id]={id,innerHTML:'',value:'',textContent:'',style:{},dataset:{},classList:{add(){},remove(){},toggle(){}},
 querySelectorAll(){return[]},querySelector(){return{}},scrollIntoView(){},files:[],set onclick(f){},set oninput(f){},set onchange(f){}});
global.document={getElementById:mk,querySelectorAll:()=>[]};global.window={};global.alert=()=>{};
global.fetch=()=>Promise.resolve({json:()=>Promise.resolve({server_keys:{}}),text:()=>Promise.resolve('')});
const src=fs.readFileSync(process.env.PAGE_JS,'utf8');
eval(src+`;global.__t={show,fail,table,get cur(){return cur}}`);
let n=0;const ck=(c,m)=>{if(!c){console.log('FAIL:',m);process.exitCode=1}else n++};
const res=jobs.ok.result;global.__t.show(res);let h=els.out.innerHTML;
ck(h.includes('The short story'),'short story');ck(h.includes('Key numbers'),'key numbers');ck(h.includes('Why might the AIs disagree'),'reasons');ck(h.includes('Technical details'),'technical details');ck(h.includes('Example from your data'),'real examples in metric cards');ck(h.includes('class="tip"'),'tooltips');ck(h.includes('Pairwise disagreement'),'pairwise');ck(h.includes('Pairwise matrix'),'matrix');
ck(h.includes('Examples by category'),'examples');ck(h.includes('State trace of this run'),'trace');ck(h.includes('data-state="QUERYING"'),'svg embedded');
ck(!h.includes('SIMULATED RUN'),'real run must not show simulated banner');ck(h.includes('/api/export/'+res.run_id+'/responses.csv'),'download link');
ck(h.includes('openai:s')&&h.includes('gemini:s'),'model names');ck(h.includes('Most agree, one differs'),'plain category label');ck(!/majority_outlier/.test(h.replace(/data-[a-z]+="[^"]*"/g,'')),'no raw technical category names shown');ck(h.includes('42'),'model responses shown');
els.flt.value='disagreement';els.srch.value='';global.__t.table();ck(els.itbl.innerHTML.includes('What is 6 times 7'),'item table rows');
global.__t.show(jobs.sample.result);h=els.out.innerHTML;ck(/unverified/.test(JSON.stringify(jobs.sample.result.comparison.overall))&&true,'sample overall');ck(h.includes('SIMULATED RUN'),'mock run shows simulated banner');ck(h.includes('Comparison with human labels'),'human comparison card');
global.__t.fail(jobs.bad);ck(els.err.textContent.includes('Nothing was simulated'),'failure advice');ck(els.out.innerHTML.includes('Where the run stopped'),'failure diagram');
ck(!els.err.textContent.includes('sk-bad-key-0000000000'),'key not shown in error');
// XSS: hostile prompt text must be escaped
const evil=JSON.parse(JSON.stringify(res));evil.comparison.items[0].prompt='<img src=x onerror=alert(1)>';evil.comparison.items[0].answers['openai:s'].text='<script>bad()</script>';
global.__t.show(evil);global.__t.table();ck(!/<img src=x/.test(els.itbl.innerHTML)&&!/<script>bad/.test(els.itbl.innerHTML),'html escaped');
console.log('front-end smoke checks passed:',n);
