from dataclasses import dataclass, field, asdict
from typing import Dict, Sequence, Optional,List
from tool_server.tf_eval.utils.log_utils import get_logger
from ...utils.utils import *
from PIL import Image
import ast
import json
import re
logger = get_logger(__name__)

def _normalize_action(action):
    if not isinstance(action, dict):
        return None
    name = action.get("name")
    arguments = action.get("arguments")
    if name == "Termination":
        name = "Terminate"
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}

def _coerce_actions(obj):
    if isinstance(obj, dict):
        if "actions" in obj:
            actions = obj["actions"]
            if isinstance(actions, dict):
                actions = [actions]
            if isinstance(actions, list):
                return [_normalize_action(action) for action in actions if _normalize_action(action)]
            return None
        if "name" in obj and "arguments" in obj:
            action = _normalize_action(obj)
            return [action] if action else None
        if "answer" in obj and obj.get("answer") not in (None, ""):
            return [{"name": "Terminate", "arguments": {"ans": str(obj["answer"])}}]
    if isinstance(obj, list) and obj:
        actions = [_normalize_action(action) for action in obj if _normalize_action(action)]
        return actions or None
    return None

def _parse_actions_full(candidate):
    candidate = candidate.strip()
    if not candidate:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            actions = _coerce_actions(parser(candidate))
            if actions:
                return actions
        except Exception:
            continue
    return None

def parse_actions(text: str):
    """Parse model output into one or more tool actions using step-compatible rules."""
    if not isinstance(text, str):
        actions = _coerce_actions(text)
        return actions, None if actions else "No action found."

    actions = _parse_actions_full(text)
    if actions:
        return actions, None

    code_blocks = re.findall(r"```\s*(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    for block in code_blocks:
        actions = _parse_actions_full(block)
        if actions:
            return actions, None

    decoder = json.JSONDecoder()
    for match in re.finditer(r"[\{\[]", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
            actions = _coerce_actions(obj)
            if actions:
                return actions, None
        except Exception:
            continue

    if text.strip():
        return None, "Invalid action format."
    return None, "No action found."

def parse_single_action(text: str):
    """Return the first executable action; preserve multi-action as a warning."""
    actions, error = parse_actions(text)
    if not actions:
        return None, error
    if len(actions) > 1:
        return actions[0], "Multiple_tool_calls_in_single_step"
    return actions[0], None

def normalize_action_response(action, thought=""):
    payload = {
        "thought": str(thought or "")[:200],
        "actions": [action],
    }
    return json.dumps(payload, ensure_ascii=False)

@dataclass
class DynamicBatchItem:
    max_rounds: int
    current_round : int
    status: str = "pending" # pending, processing, finished
    finish_reason: str = ""
    meta_data: Dict = field(default = None)
    conversation: object = field(default = None)
    model_response: List[str] = field(default_factory=list)
    tool_cfg :  List[str] = field(default_factory=list)
    tool_response :  List[str] = field(default_factory=list)
    action_parse_error : List[str] = field(default_factory=list)
    action_parse_warning : List[str] = field(default_factory=list)
    new_round_input :  List[str] = field(default_factory=list)
    gpkg : str = field(default_factory=str)
    img : str = field(default_factory=str)
    cached_gpkg : str = field(default_factory=str)
    cached_layers: Dict[str, str] = field(default_factory=dict)

class DynamicBatchManager():
    def __init__(
        self,
        batch_size: int,
        stop_token: str = "<stop>",
        max_rounds: int = 3,
        generate_conversation_fn = None,
    ):
        self.dynamic_batch = []
        self.batch_size = batch_size
        self.stop_token = stop_token
        self.max_rounds = max_rounds
        self.generate_conversation_fn = generate_conversation_fn
    
    def extract_action(self, text: str):
        action, _ = parse_single_action(text)
        return [action] if action is not None else None
        
    def extract_final_answer(self, final_response: str):
        res = ""
        action, _ = parse_single_action(final_response)
        if action and action.get("name") == "Terminate" and "ans" in action["arguments"]:
            ans = action["arguments"]["ans"]
            return ans.strip() if isinstance(ans, str) else ans
        return res
    
    def check_terminate(self, response: str):
        action, _ = parse_single_action(response)
        return bool(action and action.get("name") == "Terminate")
    
    def pop_qualified_items(self):
        res = []
        new_batch = []
        for idx,item in enumerate(self.dynamic_batch):
            if item.status == "finished":
                item = asdict(item)       
                final_model_output = item["model_response"][-1]
                final_answer = self.extract_final_answer(final_model_output)
                item["final_answer"] = final_answer
                
                res.append(item)
            else:
                new_batch.append(item)
        self.dynamic_batch = new_batch
        return res
    
    def append_item(self, meta_data: Dict):
        if len(self.dynamic_batch) < self.batch_size:
            candidate_item = DynamicBatchItem(
                max_rounds=self.max_rounds,
                current_round=0,
                meta_data=meta_data,
                status="pending"
            )
            candidate_item.conversation = self.generate_conversation_fn(
                text = meta_data["text"], 
                role = "user"
            )
            
            self.dynamic_batch.append(candidate_item)
        else:
            raise ValueError("Batch is full")
    
    def append_item_to_full(self, dataloader, progress_bar=None):
        while len(self.dynamic_batch) < self.batch_size:
            try:
                self.append_item(next(dataloader))
                if progress_bar:
                    progress_bar.update(1)
            except:
                break
        
    def get_current_batch(self):
        return self.dynamic_batch
    
    # Caution: Call this only after model.generate
    def update_item_status(self):
        for item in self.dynamic_batch:
            if item.status == "pending":
                if self.check_terminate(item.model_response[-1]):
                    item.status = "finished"
                    item.finish_reason = "terminate"
                elif item.current_round == item.max_rounds:
                    item.status = "finished"
                    item.finish_reason = "max_rounds"
                else:
                    item.current_round += 1
                    item.status = "processing"
            elif item.status == "processing":
                if self.check_terminate(item.model_response[-1]):
                    item.status = "finished"
                    item.finish_reason = "terminate"
                elif item.current_round == item.max_rounds:
                    item.status = "finished"
                    item.finish_reason = "max_rounds"
                else:
                    item.current_round += 1
            elif item.status == "finished":
                pass
            else:
                raise ValueError(f"Invalid status {item.status}")
