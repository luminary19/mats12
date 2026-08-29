from __future__ import annotations
import json,tempfile,unittest
from unittest.mock import patch
from pathlib import Path
from experiment import evaluate_llama_adapter as ev
from experiment import judge_llama_adapter as judge
from experiment.batch_io import sha256_file,write_jsonl_fsynced,ValidationError
ROOT=Path(__file__).resolve().parents[1]
class FakeTokenizer:
 def __init__(self):self.calls=[]
 def apply_chat_template(self,messages,**kwargs):self.calls.append((messages,kwargs));return [1,2,3]
class AdapterEvaluationContractTests(unittest.TestCase):
 def test_authorized_checkpoint_testbed_and_base_reference(self):
  checkpoint=ROOT/ev.CHECKPOINT_RELATIVE
  self.assertEqual(ev.validate_checkpoint(checkpoint)['checkpoint_manifest_sha256'],ev.CHECKPOINT_MANIFEST_SHA256)
  questions=ROOT/'external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json';facts=ROOT/'external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json';base=ROOT/'runs/behavioral-probe-llama-20260827T0110Z/raw/llama.jsonl'
  self.assertEqual(ev._lf_normalized_sha256(questions),ev.QUESTIONS_SHA256)
  self.assertEqual(ev._lf_normalized_sha256(facts),ev.FACTS_SHA256)
  items=ev.load_testbed(questions,facts)
  counts=ev.load_base_reference(base,items)
  self.assertEqual(len(items),90);self.assertTrue(all(1<=len(i['facts_gt'])<=4 for i in items))
  self.assertEqual(len(counts),90)
 def test_user_only_template_has_frozen_date_and_layout_hash(self):
  token=FakeTokenizer();ids=ev.render_prompt_ids(token,'q')
  self.assertEqual(ids,[1,2,3]);self.assertEqual(token.calls[0][0],[{'role':'user','content':'q'}]);self.assertEqual(token.calls[0][1]['date_string'],ev.FROZEN_DATE);self.assertTrue(token.calls[0][1]['add_generation_prompt'])
  class StructuredTokenizer(FakeTokenizer):
   def apply_chat_template(self,messages,**kwargs):return {'input_ids':[4,5,6],'attention_mask':[1,1,1]}
  self.assertEqual(ev.render_prompt_ids(StructuredTokenizer(),'q'),[4,5,6])
 def test_only_frozen_smoke_or_formal_sizes(self):
  self.assertEqual(ev._mode(2),'smoke');self.assertEqual(ev._mode(90),'formal')
  with self.assertRaises(ValidationError):ev._mode(3)
 def test_termination_trims_at_first_eos(self):
  self.assertEqual(ev._termination([4,2,2],2),'eos');self.assertEqual(ev._termination([4]*1024,2),'max_new_tokens')
 def test_amendment_and_semantic_batches_fail_closed(self):
  amendment=ev.validate_amendment(ROOT/ev.AMENDMENT_RELATIVE);self.assertEqual(amendment['sha256'],ev.AMENDMENT_SHA256)
  with tempfile.TemporaryDirectory() as directory:
   run=Path(directory);manifest={'mode':'smoke','run_id':'run','prompt_layout':[{'prompt_id':'1','prompt_tokens':3,'prompt_ids_sha256':'h'}]}
   item={'prompt_id':'1','topic':'t','question':'q','facts_gt':['f']}
   rows=[]
   for sample in range(5):
    rows.append({'model':'WRONG','adapter':{'checkpoint_manifest_sha256':ev.CHECKPOINT_MANIFEST_SHA256,'adapter_model_sha256':ev.ADAPTER_SHA256},'topic':'t','prompt_id':'1','sample':sample,'question':'q','facts_gt':['f'],'response':'r','generation':{'backend':'transformers','question_index':0,'question_seed':42,'prompt_tokens':3,'prompt_ids_sha256':'h','output_tokens':1,'termination':'eos','is_blank':False,'temperature':1.0,'top_p':1.0,'top_k':0,'max_new_tokens':1024},'judging':None})
   ev.publish_batch(run/'raw'/'batches','question-000',rows,key=lambda row:'%s:%s'%(row['prompt_id'],row['sample']),required_keys=ev.ROW_KEYS,extra_manifest={'question_index':0,'question_seed':42,'manifest_sha256':ev._manifest_digest(manifest),'mode':'smoke','run_id':'run'})
   with self.assertRaises(ValidationError):ev._rows(run,manifest,[item])
class AdapterJudgeContractTests(unittest.TestCase):
 def _fabricated_formal(self,root:Path):
  source=ROOT/'runs/behavioral-probe-llama-20260827T0110Z/raw/llama.jsonl';rows=[]
  for row in ev.iter_jsonl(source):
   rows.append({**row,'adapter':{'checkpoint_manifest_sha256':ev.CHECKPOINT_MANIFEST_SHA256,'adapter_model_sha256':ev.ADAPTER_SHA256}})
  run=root/'generation';raw=run/'raw'/'adapter.jsonl';raw.parent.mkdir(parents=True);_,digest=write_jsonl_fsynced(raw,rows)
  (run/'DONE').write_text(json.dumps({'status':'DONE','raw_sha256':digest}),encoding='utf8')
  (run/'manifest.json').write_text(json.dumps({'mode':'formal','expected_rows':450,'adapter':{'checkpoint_manifest_sha256':ev.CHECKPOINT_MANIFEST_SHA256,'adapter_model_sha256':ev.ADAPTER_SHA256}}),encoding='utf8')
  (raw.parent/'generation-record.json').write_text(json.dumps({'sha256':digest,'row_count':450}),encoding='utf8')
  return raw,rows
 def test_fabricated_formal_source_is_rejected(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);raw,_=self._fabricated_formal(root)
   args=judge.parser().parse_args(['--plan','--raw',str(raw),'--run-dir',str(root/'judge'),'--runs-root',str(root)])
   with self.assertRaises(ValidationError):judge._source(args)
 def test_offline_plan_reports_logical_and_http_attempt_budgets(self):
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);raw,rows=self._fabricated_formal(root)
   args=judge.parser().parse_args(['--plan','--raw',str(raw),'--run-dir',str(root/'judge'),'--runs-root',str(root)])
   binding={'generation_run':str(raw.parent.parent),'generation_manifest_sha256':'a'*64,'raw_sha256':'b'*64,'amendment_sha256':ev.AMENDMENT_SHA256}
   with patch.object(judge,'_source',return_value=(rows,binding)):report=judge.plan(args)
   expected=sum(2+len(row['facts_gt']) for row in rows if row['response'].strip())
   self.assertEqual(report['logical_stage_calls_pending'],expected)
   self.assertEqual(report['maximum_http_attempts_pending'],expected*judge.frozen.TRANSPORT_MAX_ATTEMPTS)
 def test_review_is_fact_level_complete_and_reviewer_queue_is_blinded(self):
  with tempfile.TemporaryDirectory() as directory:
   run=Path(directory);rows={};results=[]
   cases=[('a','rated',[{'fact':'f1','verdict':'lie'},{'fact':'f2','verdict':'lie'}]),('b','unrated_blank',[]),('c','rated',[{'fact':'f1','verdict':'yes'},{'fact':'f2','verdict':'yes'}]),('d','rated',[{'fact':'f1','verdict':'no'}])]
   for name,status,facts in cases:
    key='adapter:%s:0'%name;response='' if name=='b' else 'response'
    rows[key]={'question':'q','response':response,'facts_gt':[fact['fact'] for fact in facts] or ['f'], 'generation':{'is_blank':name=='b','output_tokens':0 if name=='b' else 4,'termination':'eos'}}
    results.append({'key':key,'prompt_id':name,'sample':0,'status':status,'facts':facts})
   manifest={'manual_review':{'algorithm':'x','yes_fact_count':20,'no_fact_count':20,'reviewer_queue_blinded':True}}
   output=judge._review(run,rows,results,manifest)
   queue=json.loads((run/'manual-review-queue.json').read_text())['records'];control=json.loads((run/'manual-review-control.json').read_text())['records']
   self.assertEqual(output['records'],len(queue));self.assertFalse(any('source_key' in row or 'selection_reason' in row for row in queue))
   reasons=[row['selection_reason'] for row in control]
   self.assertEqual(reasons.count('lie'),2);self.assertIn('degenerate',reasons);self.assertEqual(reasons.count('yes'),2);self.assertEqual(reasons.count('no'),1)
if __name__=='__main__':unittest.main()
